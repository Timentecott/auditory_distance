"""
Calibration script for experiment 2 using short_tap stimuli.
Based on the loudness calibration function from experiment_1.py
"""

from psychopy import visual, event, core
import numpy as np
import os
os.environ["SD_ENABLE_ASIO"] = "1"  # this line is important as it allows revelation of asio devices
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import threading


def ensure_stereo(audio):
    """Force audio to stereo (L/R) for headphone playback."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.column_stack([audio[:, 0], audio[:, 0]])
    return audio[:, :2]


def route_to_asio_channels(audio, device_role):
    """Route stereo audio to ASIO channel 1 for headphones or channels 3-4 for loudspeaker."""
    audio = ensure_stereo(np.asarray(audio))
    routed = np.zeros((audio.shape[0], 4), dtype=np.float32)

    if device_role in ['in_situ_headphone', 'ex_situ_headphone']:
        routed[:, 2:4] = audio[:, :2]   # ASIO channels 3-4
    elif device_role == 'speaker':
        routed[:, 0:2] = audio[:, :2]   # ASIO channels 1-2
    else:
        raise ValueError(f"Unknown device_role: {device_role}")
    return routed


def resolve_output_sample_rate(device_index, preferred_rate=48000, channels=4, dtype='float32'):
    """Pick a sample rate that the output device actually supports."""
    device_info = sd.query_devices(device_index, 'output')
    candidate_rates = []
    for rate in [preferred_rate, device_info.get('default_samplerate'), 44100, 48000, 88200, 96000, 32000, 22050]:
        if rate is None:
            continue
        rate = int(round(rate))
        if rate not in candidate_rates:
            candidate_rates.append(rate)

    last_error = None
    for rate in candidate_rates:
        try:
            sd.check_output_settings(device=device_index, samplerate=rate, channels=channels, dtype=dtype)
            return rate
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"No supported output sample rate found for device {device_index} ({device_info['name']}): {last_error}"
    )


def run_short_tap_calibration(win, headphones_device, speakers_device, ASIO_AGGREGATE_DEVICE, sample_rate=None):
    """Play loudspeaker continuously, then alternate headphone and loudspeaker short_tap sounds every second."""

    headphone_file = Path(r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\localised\short_tap_in-situ-near.wav")
    speaker_file = Path(r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\short_tap.wav")

    if headphones_device != speakers_device:
        print(f"Warning: calibration will use device {speakers_device} for both speaker and headphone routing.")

    if headphones_device != ASIO_AGGREGATE_DEVICE or speakers_device != ASIO_AGGREGATE_DEVICE:
        print(f"Warning: ASIO channel routing is configured for device index {ASIO_AGGREGATE_DEVICE}.")

    def _load_audio_file(audio_path):
        audio, sr = sf.read(str(audio_path), dtype='float32', always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr)

    if not headphone_file.exists():
        raise FileNotFoundError(f"Missing headphone calibration file: {headphone_file}")
    if not speaker_file.exists():
        raise FileNotFoundError(f"Missing loudspeaker calibration file: {speaker_file}")

    headphone_audio, headphone_sr = _load_audio_file(headphone_file)
    speaker_audio, speaker_sr = _load_audio_file(speaker_file)

    if headphone_sr != speaker_sr:
        raise ValueError(f"Calibration audio files must share the same sample rate: [{headphone_sr}, {speaker_sr}]")
    if sample_rate is None:
        sample_rate = headphone_sr
    elif int(sample_rate) != headphone_sr:
        raise ValueError(f"Calibration sample rate {sample_rate} does not match audio file sample rate {headphone_sr}.")
    sample_rate = int(sample_rate)

    one_second = sample_rate 
    five_seconds = sample_rate * 5

    # Use full audio or truncate to available length
    headphone_segment = route_to_asio_channels(headphone_audio[:min(len(headphone_audio), one_second)], 'in_situ_headphone')
    speaker_segment_continuous = route_to_asio_channels(speaker_audio[:min(len(speaker_audio), five_seconds)], 'speaker')
    speaker_segment = route_to_asio_channels(speaker_audio[:min(len(speaker_audio), one_second)], 'speaker')

    if headphone_segment.shape[0] == 0 or speaker_segment.shape[0] == 0:
        raise ValueError("Calibration audio files must contain at least one second of audio.")

    def make_loop_state(audio):
        return {'audio': audio.astype(np.float32), 'pos': 0}

    def make_loop_callback(state, lock):
        def _callback(outdata, frame_count, time_info, status):
            with lock:
                audio = state['audio']
                pos = state['pos']
                end_pos = pos + frame_count
                if end_pos <= audio.shape[0]:
                    outdata[:] = audio[pos:end_pos]
                    state['pos'] = end_pos % audio.shape[0]
                else:
                    first = audio[pos:]
                    remaining = frame_count - len(first)
                    second = audio[:remaining]
                    outdata[:len(first)] = first
                    outdata[len(first):] = second
                    state['pos'] = remaining % audio.shape[0]
        return _callback

    speaker_state = make_loop_state(speaker_segment_continuous)
    speaker_lock = threading.Lock()
    speaker_text = visual.TextStim(
        win,
        text=(
            "Short Tap Calibration Preview\n\n"
            "You will now hear a continuous loudspeaker short tap sound.\n"
            "Press any key to switch to the alternating calibration."
        ),
        color='white',
        height=30,
        wrapWidth=1100
    )
    speaker_text.draw()
    win.flip()

    with sd.OutputStream(
        samplerate=sample_rate,
        device=speakers_device,
        channels=4,
        dtype='float32',
        callback=make_loop_callback(speaker_state, speaker_lock),
        latency='low',
    ):
        while True:
            if event.getKeys():
                break
            speaker_text.draw()
            win.flip()
            core.wait(0.01)


if __name__ == '__main__':
    # Example usage:
    # Initialize a PsychoPy window
    win = visual.Window([1920, 1080], fullscr=False, monitor='testMonitor')

    # Configure your ASIO devices (modify these values based on your setup)
    ASIO_AGGREGATE_DEVICE = 10  # Change this to your ASIO aggregate device index
    headphones_device = ASIO_AGGREGATE_DEVICE
    speakers_device = ASIO_AGGREGATE_DEVICE

    try:
        run_short_tap_calibration(win, headphones_device, speakers_device, ASIO_AGGREGATE_DEVICE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        win.close()
        core.quit()
