# Standalone calibration script extracted from experiment.py

from psychopy import visual, event, core
import os
os.environ["SD_ENABLE_ASIO"] = "1"
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal
import threading
from pathlib import Path

# Audio device indices (same as experiment.py)
ASIO_AGGREGATE_DEVICE = 16  # ASIO4ALL v2 aggregate: 4 output channels
sample_rate = 48000

INTRO_SPEAKER_FILE = Path(r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\loudspeaker_stimuli_bob\noise\brown_noise_5s.wav")
HEADPHONE_FILE = Path(r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\ex_situ_stimuli_bob\noise\brown_noise_5s.wav")
SPEAKER_FILE = Path(r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\loudspeaker_stimuli_bob\noise\brown_noise_5s.wav")


def apply_fade(audio, sample_rate, fade_ms=10):
    """Apply short fade-in/out to reduce clicks at playback boundaries."""
    fade_samples = int(sample_rate * fade_ms / 1000.0)
    if fade_samples <= 0:
        return audio

    n_samples = audio.shape[0]
    if n_samples < 2 * fade_samples:
        fade_samples = n_samples // 2
    if fade_samples <= 0:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=True)
    fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=True)

    audio = audio.copy()
    if audio.ndim == 1:
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    else:
        audio[:fade_samples, :] *= fade_in[:, None]
        audio[-fade_samples:, :] *= fade_out[:, None]
    return audio


def ensure_stereo(audio):
    """Force audio to stereo (L/R) for headphone playback."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.column_stack([audio[:, 0], audio[:, 0]])
    return audio[:, :2]


def route_to_asio_channels(audio, device_role):
    """Route stereo audio to ASIO channel 1 for loudspeaker or channels 3-4 for headphone."""
    audio = ensure_stereo(np.asarray(audio))
    routed = np.zeros((audio.shape[0], 4), dtype=np.float32)
    if device_role == 'speaker':
        routed[:, 0] = audio[:, 0]
    elif device_role in ['in_situ_headphone', 'ex_situ_headphone']:
        routed[:, 2:4] = audio[:, :2]
    else:
        raise ValueError(f"Unknown device_role: {device_role}")
    return routed


def _load_audio_file(audio_path):
    audio, sr = sf.read(str(audio_path), dtype='float32', always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        n_samples = int(round(audio.shape[0] * (sample_rate / sr)))
        if audio.ndim == 1:
            audio = signal.resample(audio, n_samples)
        else:
            audio = np.column_stack([signal.resample(audio[:, ch], n_samples) for ch in range(audio.shape[1])])
    return audio, sample_rate


def _first_second(audio):
    return audio[:sample_rate]


def run_loudness_calibration(win, headphones_device, speakers_device, sample_rate=48000):
    """Calibrate headphone level against a fixed speaker reference using on-screen buttons."""
    if headphones_device != speakers_device:
        print(f"Warning: calibration will use device {speakers_device} for both speaker and headphone routing.")

    if headphones_device != ASIO_AGGREGATE_DEVICE or speakers_device != ASIO_AGGREGATE_DEVICE:
        print(f"Warning: ASIO channel routing is configured for device index {ASIO_AGGREGATE_DEVICE}.")

    if not INTRO_SPEAKER_FILE.exists():
        raise FileNotFoundError(f"Missing intro loudspeaker file: {INTRO_SPEAKER_FILE}")
    if not HEADPHONE_FILE.exists():
        raise FileNotFoundError(f"Missing headphone calibration file: {HEADPHONE_FILE}")
    if not SPEAKER_FILE.exists():
        raise FileNotFoundError(f"Missing loudspeaker calibration file: {SPEAKER_FILE}")

    intro_audio, _ = _load_audio_file(INTRO_SPEAKER_FILE)
    headphone_audio_full, _ = _load_audio_file(HEADPHONE_FILE)
    speaker_audio_full, _ = _load_audio_file(SPEAKER_FILE)

    intro_audio = apply_fade(ensure_stereo(_first_second(intro_audio)), sample_rate, fade_ms=20)
    headphone_base = apply_fade(ensure_stereo(_first_second(headphone_audio_full)), sample_rate, fade_ms=20)
    speaker_base = apply_fade(ensure_stereo(_first_second(speaker_audio_full)), sample_rate, fade_ms=20)

    headphone_offset_db = 0.0
    step_db = 1.0
    calibration_log = []

    win.flip()

    intro_text = visual.TextStim(
        win,
        text=(
            "Loudspeaker preview\n\n"
            "You will first hear continuous brown noise from the loudspeaker.\n"
            "Press any key when ready to start the calibration."
        ),
        color='white',
        height=30,
        wrapWidth=1100
    )
    intro_text.draw()
    win.flip()

    intro_state = {
        'audio': route_to_asio_channels(intro_audio, 'speaker'),
        'pos': 0,
    }
    intro_lock = threading.Lock()

    def intro_callback(outdata, frame_count, time_info, status):
        with intro_lock:
            audio = intro_state['audio']
            pos = intro_state['pos']
            end_pos = pos + frame_count
            if end_pos <= audio.shape[0]:
                outdata[:] = audio[pos:end_pos]
                intro_state['pos'] = end_pos % audio.shape[0]
            else:
                first = audio[pos:]
                remaining = frame_count - len(first)
                second = audio[:remaining]
                outdata[:len(first)] = first
                outdata[len(first):] = second
                intro_state['pos'] = remaining

    with sd.OutputStream(
        samplerate=sample_rate,
        device=speakers_device,
        channels=4,
        dtype='float32',
        callback=intro_callback,
        latency='low',
    ):
        event.waitKeys()

    old_mouse_visible = win.mouseVisible
    win.mouseVisible = True
    mouse = event.Mouse(win=win)

    plus_box = visual.Rect(win, pos=(-320, -260), width=180, height=90, fillColor='darkgreen', lineColor='white')
    minus_box = visual.Rect(win, pos=(0, -260), width=180, height=90, fillColor='darkred', lineColor='white')
    store_box = visual.Rect(win, pos=(320, -260), width=220, height=90, fillColor='darkblue', lineColor='white')

    plus_text = visual.TextStim(win, text='+', pos=(-320, -260), color='white', height=48)
    minus_text = visual.TextStim(win, text='-', pos=(0, -260), color='white', height=48)
    store_text = visual.TextStim(win, text='Store', pos=(320, -260), color='white', height=34)

    status_text = visual.TextStim(win, text='', pos=(0, 220), color='white', height=28, wrapWidth=1200)
    info_text = visual.TextStim(win, text='', pos=(0, 160), color='white', height=24, wrapWidth=1200)
    big_one = visual.TextStim(win, text='1', pos=(-280, -20), color='white', height=80)
    big_two = visual.TextStim(win, text='2', pos=(280, -20), color='white', height=80)

    def draw_calibration_screen(message='', phase_label=''):
        plus_box.draw()
        minus_box.draw()
        store_box.draw()
        plus_text.draw()
        minus_text.draw()
        store_text.draw()
        big_one.draw()
        big_two.draw()
        status_text.setText(phase_label)
        status_text.draw()
        info_text.setText(f"Current adjustment for 1: {headphone_offset_db:+.1f} dB\n{message}")
        info_text.draw()
        win.flip()

    mouse.clickReset()
    prev_mouse_down = False
    store_selected = False

    sd.default.latency = 'low'

    playback_state = {
        'audio': None,
        'pos': 0,
    }
    state_lock = threading.Lock()

    def build_cycle_audio():
        headphone_audio = ensure_stereo(headphone_base.copy() * (10 ** (headphone_offset_db / 20.0)))
        headphone_routed = route_to_asio_channels(headphone_audio, 'in_situ_headphone')
        speaker_routed = route_to_asio_channels(speaker_base, 'speaker')
        cycle = np.concatenate([headphone_routed, speaker_routed], axis=0)
        return np.concatenate([cycle, cycle], axis=0)

    def rebuild_playback_buffer():
        with state_lock:
            playback_state['audio'] = build_cycle_audio()
            playback_state['pos'] = 0

    rebuild_playback_buffer()

    def maybe_handle_click():
        nonlocal headphone_offset_db, prev_mouse_down, store_selected
        mouse_down = mouse.getPressed()[0]
        if mouse_down and not prev_mouse_down:
            if plus_box.contains(mouse):
                headphone_offset_db += step_db
                calibration_log.append(headphone_offset_db)
                rebuild_playback_buffer()
                print(f"Sound 1 increased to {headphone_offset_db:+.1f} dB")
            elif minus_box.contains(mouse):
                headphone_offset_db -= step_db
                calibration_log.append(headphone_offset_db)
                rebuild_playback_buffer()
                print(f"Sound 1 decreased to {headphone_offset_db:+.1f} dB")
            elif store_box.contains(mouse):
                store_selected = True
        prev_mouse_down = mouse_down

    def calibration_callback(outdata, frame_count, time_info, status):
        with state_lock:
            audio = playback_state['audio']
            pos = playback_state['pos']
            if audio is None:
                outdata[:] = 0
                return
            end_pos = pos + frame_count
            if end_pos <= audio.shape[0]:
                outdata[:] = audio[pos:end_pos]
                playback_state['pos'] = end_pos
            else:
                first = audio[pos:]
                remaining = frame_count - len(first)
                second = audio[:remaining]
                outdata[:len(first)] = first
                outdata[len(first):] = second
                playback_state['pos'] = remaining

    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            device=speakers_device,
            channels=4,
            dtype='float32',
            callback=calibration_callback,
            latency='low',
        ):
            while not store_selected:
                maybe_handle_click()
                draw_calibration_screen("Adjust sound 1 until it matches sound 2.", "Playing: 1 -> 2")
                core.wait(0.01)
    finally:
        sd.stop()

    done = visual.TextStim(
        win,
        text=(
            "Calibration complete.\n\n"
            f"Adjustment to apply to sound 1: {headphone_offset_db:+.1f} dB\n\n"
            "Press any key to close."
        ),
        color='white',
        height=28,
        wrapWidth=1000
    )
    done.draw()
    win.flip()
    event.waitKeys()
    win.mouseVisible = old_mouse_visible
    print(f"\nCalibration log: {calibration_log}")
    return headphone_offset_db, 0.0


def main():
    win = visual.Window(
        size=(1024, 768),
        units='pix',
        fullscr=True,
        color=(0, 0, 0),
        allowStencil=False
    )
    win.mouseVisible = False

    headphones_device = ASIO_AGGREGATE_DEVICE
    speakers_device = ASIO_AGGREGATE_DEVICE

    try:
        run_loudness_calibration(
            win,
            headphones_device=headphones_device,
            speakers_device=speakers_device,
            sample_rate=sample_rate,
        )
    finally:
        win.close()
        core.quit()


if __name__ == '__main__':
    main()
