import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

try:
    import msvcrt
except ImportError:
    msvcrt = None


# Audio configuration
SAMPLE_RATE = 48000
BLOCKSIZE = 1024
TEST_AMPLITUDE = 0.08
NOISE_BUFFER_SECONDS = 120
REF_PRESSURE = 20e-6
MIC_SENSITIVITY_DBV_PA = -46.0
CALIBRATION_OFFSET_DB = 0.0

# Meter configuration
LEQ_WINDOW_SEC = 1.0
SMOOTHING_TAU_SEC = 0.5


# Live SPL state
_latest_spl_db = float("nan")
_smoothed_spl_db = float("nan")
_latest_dbfs = float("nan")

# History of (mean_square, n_samples) for rolling Leq
_ms_history = deque()
_samples_in_window = 0

_spl_lock = threading.Lock()


def list_devices():
    """Print available audio devices with indices and channel counts."""
    print("\n=== Available Audio Devices ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(
            f"{i}: {dev['name']} | "
            f"In={dev['max_input_channels']} Out={dev['max_output_channels']} "
            f"DefaultSR={dev['default_samplerate']}"
        )


def calculate_spl(audio_rms, mic_sensitivity_dbv_pa=MIC_SENSITIVITY_DBV_PA):
    """Convert RMS signal level to SPL estimate using microphone sensitivity."""
    if not np.isfinite(audio_rms) or audio_rms <= 0:
        return float("nan")

    voltage = audio_rms
    mic_sens_linear = 10 ** (mic_sensitivity_dbv_pa / 20.0)
    pressure_pa = voltage / mic_sens_linear

    if pressure_pa <= 0:
        return float("nan")

    return 20 * np.log10(pressure_pa / REF_PRESSURE) + CALIBRATION_OFFSET_DB


def rms_to_dbfs(audio_rms):
    if not np.isfinite(audio_rms) or audio_rms <= 0:
        return float("nan")
    return 20 * np.log10(audio_rms)


def _compute_window_leq_locked():
    if _samples_in_window <= 0 or len(_ms_history) == 0:
        return float("nan"), float("nan")

    weighted_ms = 0.0
    for ms, n in _ms_history:
        weighted_ms += ms * n

    mean_square = weighted_ms / _samples_in_window
    if mean_square <= 0 or not np.isfinite(mean_square):
        return float("nan"), float("nan")

    rms = float(np.sqrt(mean_square))
    return calculate_spl(rms), rms_to_dbfs(rms)


def _input_callback(indata, frames, time_info, status):
    del time_info
    if status:
        print(f"\nInput status: {status}")

    mono = np.asarray(indata[:, 0], dtype=np.float64)
    mean_square = float(np.mean(mono * mono))
    rms = float(np.sqrt(mean_square))
    spl = calculate_spl(rms)
    dbfs = rms_to_dbfs(rms)

    block_sec = max(1e-6, frames / SAMPLE_RATE)
    alpha = 1.0 - np.exp(-block_sec / SMOOTHING_TAU_SEC)

    with _spl_lock:
        global _latest_spl_db, _smoothed_spl_db, _samples_in_window, _latest_dbfs

        _latest_spl_db = spl
        _latest_dbfs = dbfs

        if np.isfinite(spl):
            if np.isfinite(_smoothed_spl_db):
                _smoothed_spl_db = (1.0 - alpha) * _smoothed_spl_db + alpha * spl
            else:
                _smoothed_spl_db = spl

        _ms_history.append((mean_square, frames))
        _samples_in_window += frames

        max_samples = int(LEQ_WINDOW_SEC * SAMPLE_RATE)
        while _samples_in_window > max_samples and len(_ms_history) > 0:
            old_ms, old_n = _ms_history.popleft()
            _samples_in_window -= old_n


def _make_noise_generator(fs, amplitude):
    """Return a callback that outputs stable white noise continuously."""
    rng = np.random.default_rng()
    n_samples = int(fs * NOISE_BUFFER_SECONDS)

    # High-quality white noise buffer
    noise = rng.standard_normal(n_samples).astype(np.float32)
    noise = noise - np.mean(noise)  # remove DC offset

    # Set stable global RMS to requested amplitude
    rms = float(np.sqrt(np.mean(noise * noise)))
    if rms > 0:
        noise = noise * (amplitude / rms)

    read_pos = 0

    def output_callback(outdata, frames, time_info, status):
        nonlocal read_pos
        del time_info
        if status:
            print(f"\nOutput status: {status}")

        if read_pos + frames <= n_samples:
            y = noise[read_pos:read_pos + frames]
            read_pos += frames
        else:
            first = noise[read_pos:]
            remain = frames - len(first)
            second = noise[:remain]
            y = np.concatenate((first, second), axis=0)
            read_pos = remain

        if outdata.shape[1] == 1:
            outdata[:, 0] = y
        else:
            outdata[:] = np.repeat(y[:, np.newaxis], outdata.shape[1], axis=1)

    return output_callback


def _space_pressed_nonblocking():
    """Return True when space is pressed (Windows console)."""
    if msvcrt is None:
        return False
    if not msvcrt.kbhit():
        return False

    ch = msvcrt.getwch()
    if ch == " ":
        return True
    return False


def _poll_nonblocking_key():
    """Read one non-blocking key from Windows console, else None."""
    if msvcrt is None or not msvcrt.kbhit():
        return None
    return msvcrt.getwch().lower()


def run_test(output_device, input_device):
    """Play noise through output while measuring live SPL at the microphone until space is pressed."""
    global _latest_spl_db, _smoothed_spl_db, _samples_in_window, _latest_dbfs
    with _spl_lock:
        _latest_spl_db = float("nan")
        _smoothed_spl_db = float("nan")
        _latest_dbfs = float("nan")
        _ms_history.clear()
        _samples_in_window = 0

    out_info = sd.query_devices(output_device)
    in_info = sd.query_devices(input_device)

    out_channels = min(2, int(out_info["max_output_channels"]))
    if out_channels < 1:
        raise ValueError(f"Device {output_device} has no output channels")

    in_channels = 1
    if int(in_info["max_input_channels"]) < 1:
        raise ValueError(f"Device {input_device} has no input channels")

    output_callback = _make_noise_generator(SAMPLE_RATE, TEST_AMPLITUDE)
    reference_leq_dbfs = None

    print("\nStarting test...")
    print(f"  Output device: {output_device} ({out_info['name']})")
    print(f"  Input device:  {input_device} ({in_info['name']})")
    if msvcrt is not None:
        print("  Press SPACE to stop this sound check. Press R to set current Leq as reference.")
    else:
        print("  Press Ctrl+C to stop this sound check.")
    print(f"  Meter: Inst_SPL | Smooth_SPL | Leq_SPL({LEQ_WINDOW_SEC:.1f}s) | Leq_dBFS | Delta_ref")

    start = time.time()
    with sd.OutputStream(
        device=output_device,
        samplerate=SAMPLE_RATE,
        channels=out_channels,
        dtype="float32",
        callback=output_callback,
        blocksize=BLOCKSIZE,
    ), sd.InputStream(
        device=input_device,
        samplerate=SAMPLE_RATE,
        channels=in_channels,
        dtype="float32",
        callback=_input_callback,
        blocksize=BLOCKSIZE,
    ):
        try:
            while True:
                elapsed = time.time() - start

                with _spl_lock:
                    inst = _latest_spl_db
                    smooth = _smoothed_spl_db
                    leq_spl, leq_dbfs = _compute_window_leq_locked()

                inst_str = f"{inst:7.2f}" if np.isfinite(inst) else "   ... "
                smooth_str = f"{smooth:7.2f}" if np.isfinite(smooth) else "   ... "
                leq_spl_str = f"{leq_spl:7.2f}" if np.isfinite(leq_spl) else "   ... "
                leq_dbfs_str = f"{leq_dbfs:8.2f}" if np.isfinite(leq_dbfs) else "    ... "

                if reference_leq_dbfs is not None and np.isfinite(leq_dbfs):
                    delta = leq_dbfs - reference_leq_dbfs
                    delta_str = f"{delta:+6.2f} dB"
                else:
                    delta_str = "  n/a  "

                print(
                    f"\r  t={elapsed:6.1f}s | Inst={inst_str} | Smooth={smooth_str} | "
                    f"Leq={leq_spl_str} | dBFS={leq_dbfs_str} | Δ={delta_str}",
                    end="",
                )

                key = _poll_nonblocking_key()
                if key == " ":
                    break
                if key == "r" and np.isfinite(leq_dbfs):
                    reference_leq_dbfs = leq_dbfs
                    print(f"\nReference set: {reference_leq_dbfs:.2f} dBFS")

                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    print("\r  Test finished.                                                                 ")


def ask_index(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer index.")


def main():
    print("=== Headphone / Loudspeaker SPL Check ===")
    list_devices()

    headphone_output = ask_index("\nEnter headphone output device index: ")
    loudspeaker_output = ask_index("Enter loudspeaker output device index: ")
    input_device = ask_index("Enter microphone input device index: ")

    while True:
        print("\nSelect action:")
        print("  [H] Test headphone output")
        print("  [L] Test loudspeaker output")
        print("  [Q] Quit")

        choice = input("Choice: ").strip().lower()

        if choice == "h":
            run_test(headphone_output, input_device)
        elif choice == "l":
            run_test(loudspeaker_output, input_device)
        elif choice == "q":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Enter H, L, or Q.")


if __name__ == "__main__":
    main()


