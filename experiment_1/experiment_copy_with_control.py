#this is the same as pilot experiment but with the perceptual loudness calibration i'm working on

from psychopy import visual, event, core
import pandas as pd
import time
import numpy as np
import random
import os, glob
os.environ["SD_ENABLE_ASIO"] = "1" #this line is important as it allows revelation of asio devices
import sounddevice as sd

import soundfile as sf

from pathlib import Path
from scipy import signal
import threading

# Personal EQ is disabled for now because there is no local EQ file available.


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


def append_silence_tail(audio, sample_rate, tail_ms=20):
    """Append a short silence tail to avoid device-end clicks on some outputs."""
    tail_samples = int(sample_rate * tail_ms / 1000.0)
    if tail_samples <= 0:
        return audio

    if audio.ndim == 1:
        tail = np.zeros(tail_samples, dtype=audio.dtype)
    else:
        tail = np.zeros((tail_samples, audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, tail], axis=0)


def play_audio_on_stream(audio, stream):
    """Play a block of audio through an already-open output stream."""
    audio = np.asarray(audio, dtype=np.float32)
    stream.write(audio)


def play_audio_on_device(audio, sample_rate, device_index, mapping=None):
    """Play a block of audio through a device using the same ASIO path as the device test script."""
    audio = np.asarray(audio, dtype=np.float32)
    print(f"  [play_audio_on_device] device={device_index}, mapping={mapping}, shape={audio.shape}, dtype={audio.dtype}")
    sd.play(audio, samplerate=sample_rate, device=device_index, mapping=mapping)
    sd.wait()
    print(f"  [play_audio_on_device] playback complete")


def apply_gain_db(audio, gain_db):
    """Apply gain in dB to an audio array."""
    return audio * (10 ** (gain_db / 20.0))


def apply_bandpass_filter(audio, sample_rate, hp_freq=80, lp_freq=10000, order=4):
    """
    Apply high-pass and low-pass filters to reduce headphone/speaker differences.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        hp_freq: High-pass cutoff frequency in Hz (default 80 Hz)
        lp_freq: Low-pass cutoff frequency in Hz (default 10000 Hz)
        order: Filter order (default 4 for gentle slope)
    
    Returns:
        Filtered audio array
    """
    if len(audio) == 0:
        return audio
    
    # High-pass filter (remove low-frequency rumble/bass)
    sos_hp = signal.butter(order, hp_freq, btype='high', fs=sample_rate, output='sos')
    audio = signal.sosfilt(sos_hp, audio, axis=0)
    
    # Low-pass filter (remove high-frequency hiss/treble)
    sos_lp = signal.butter(order, lp_freq, btype='low', fs=sample_rate, output='sos')
    audio = signal.sosfilt(sos_lp, audio, axis=0)
    
    return audio


def apply_device_specific_filter(audio, sample_rate, device_type='headphone', order=4, apply_hp=True, apply_lp=True):
    """
    Apply device-specific EQ to match frequency response between headphones and speakers.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        device_type: 'headphone' or 'speaker'
        order: Filter order
        apply_hp: If True, apply high-pass filter (default True)
        apply_lp: If True, apply low-pass filter (default True)
    
    Returns:
        Filtered audio array
    """
    if len(audio) == 0:
        return audio
    
    if device_type == 'headphone':
        # For headphones: reduce treble (tinny sound), keep some presence
        # High-pass: 100 Hz (remove rumble)
        # Low-pass: 7000 Hz (aggressive treble reduction for tinny correction)
        hp_freq, lp_freq = 100, 7000
    else:
        # For speakers: gentler filtering, keep more bass and treble
        # High-pass: 100 Hz (less aggressive)
        # Low-pass: 7000 Hz (more treble)
        hp_freq, lp_freq = 100, 7000
    
    # High-pass filter (optional)
    if apply_hp:
        sos_hp = signal.butter(order, hp_freq, btype='high', fs=sample_rate, output='sos')
        audio = signal.sosfilt(sos_hp, audio, axis=0)
    
    # Low-pass filter (optional)
    if apply_lp:
        sos_lp = signal.butter(order, lp_freq, btype='low', fs=sample_rate, output='sos')
        audio = signal.sosfilt(sos_lp, audio, axis=0)
    
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
        routed[:, 0] = audio[:, 0]  # Left channel to ASIO channel 1 only
    elif device_role == 'headphone':
        routed[:, 2:4] = audio[:, :2]
    else:
        raise ValueError(f"Unknown device_role: {device_role}")
    return routed


def collapse_to_left_channel(audio, preserve_total_rms=True):
    """
    Collapse audio to left channel only for loudspeaker playback.

    Converts stereo/mono to stereo with left = audio, right = 0 (silence).    

    Args:
        audio: Audio array (mono or stereo)
        preserve_total_rms: If True, scale the resulting single-channel-left
            stereo so the overall RMS (across channels) matches the input's
            RMS. This preserves the total signal energy when collapsing.

    Returns:
        Stereo audio array with right channel silenced. dtype matches input.
    """
    if audio is None:
        return audio

    # compute input RMS across all elements (samples x channels)
    try:
        audio_f = audio.astype(np.float64)
    except Exception:
        audio_f = np.array(audio, dtype=np.float64)

    if audio_f.size == 0:
        # empty input
        if audio.ndim == 1:
            return np.zeros((0, 2), dtype=audio.dtype)
        return np.zeros((0, 2), dtype=audio.dtype)

    orig_rms = np.sqrt(np.mean(np.square(audio_f)))

    if audio.ndim == 1:
        # Mono: convert to stereo with left = audio, right = silence
        stereo = np.zeros((len(audio), 2), dtype=audio.dtype)
        stereo[:, 0] = audio
    elif audio.shape[1] >= 2:
        # Stereo or multi-channel: keep left (channel 0), silence right
        stereo = np.zeros_like(audio[:, :2])
        stereo[:, 0] = audio[:, 0]
    else:
        # Single channel stereo-like array: convert to stereo
        stereo = np.zeros((audio.shape[0], 2), dtype=audio.dtype)
        stereo[:, 0] = audio[:, 0]

    if preserve_total_rms:
        # compute new RMS and scale to match original RMS
        stereo_f = stereo.astype(np.float64)
        new_rms = np.sqrt(np.mean(np.square(stereo_f)))
        if new_rms > 0 and orig_rms > 0:
            scale = float(orig_rms / new_rms)
            stereo = (stereo_f * scale).astype(audio.dtype)

    return stereo

def run_loudness_calibration(win, headphones_device, speakers_device, sample_rate=48000):
    """Calibrate headphone level against a fixed speaker reference using on-screen buttons."""
    if headphones_device != speakers_device:
        print(f"Warning: calibration will use device {speakers_device} for both speaker and headphone routing.")

    if headphones_device != 16 or speakers_device != 16:
        print("Warning: ASIO channel routing is configured for device index 16.")

    rng = np.random.default_rng(42)
    duration_s = 1.0
    reference_audio = rng.standard_normal(int(sample_rate * duration_s)).astype(np.float32)
    # scale reference to target RMS for calibration
    target_rms = 0.01
    cur_rms = np.sqrt(np.mean(np.square(reference_audio.astype(np.float64))))
    if cur_rms > 0:
        reference_audio = (reference_audio.astype(np.float64) * (target_rms / cur_rms)).astype(np.float32)

    # Personal EQ is disabled, so use the raw reference audio.
    reference_headphone_eq = ensure_stereo(reference_audio).astype(np.float32)

    headphone_offset_db = 0.0
    step_db = 1.0
    calibration_log = []

    # Clear any prior screen content before showing calibration instructions.
    win.flip()

    instruction = visual.TextStim(
        win,
        text=(
            "Calibration phase\n\n"
            "You will hear a continuous alternation: 1 second loudspeaker, 1 second headphone, repeating.\n"
            "Use the buttons to adjust ONLY the headphone volume until both sound equally loud.\n\n"
            "+  increases headphone volume\n"
            "-  decreases headphone volume\n"
            "Store  saves the current headphone adjustment\n\n"
            "Press any key to begin calibration."
        ),
        color='white',
        height=28,
        wrapWidth=1100
    )
    instruction.draw()
    win.flip()
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

    status_text = visual.TextStim(
        win,
        text='',
        pos=(0, 220),
        color='white',
        height=28,
        wrapWidth=1200
    )
    info_text = visual.TextStim(
        win,
        text='',
        pos=(0, 160),
        color='white',
        height=24,
        wrapWidth=1200
    )

    def draw_calibration_screen(message='', phase_label=''):
        plus_box.draw()
        minus_box.draw()
        store_box.draw()
        plus_text.draw()
        minus_text.draw()
        store_text.draw()
        status_text.setText(phase_label)
        status_text.draw()
        info_text.setText(f"Current headphone adjustment: {headphone_offset_db:+.1f} dB\n{message}")
        info_text.draw()
        win.flip()

    def make_speaker_audio():
        audio = apply_gain_db(reference_audio, 0.0)
        audio = collapse_to_left_channel(audio, preserve_total_rms=True)
        audio = apply_fade(audio, sample_rate, fade_ms=20)
        return ensure_stereo(audio)
 
    def make_headphone_audio(current_hp_offset_db):
        audio = apply_gain_db(reference_headphone_eq, current_hp_offset_db)
        audio = apply_fade(audio, sample_rate, fade_ms=20)
        return ensure_stereo(audio)

    mouse.clickReset()
    prev_mouse_down = False
    store_selected = False

    def maybe_handle_click():
        nonlocal headphone_offset_db, prev_mouse_down, store_selected
        mouse_down = mouse.getPressed()[0]
        if mouse_down and not prev_mouse_down:
            if plus_box.contains(mouse):
                headphone_offset_db += step_db
                calibration_log.append(headphone_offset_db)
                print(f"Headphone level increased to {headphone_offset_db:+.1f} dB")
            elif minus_box.contains(mouse):
                headphone_offset_db -= step_db
                calibration_log.append(headphone_offset_db)
                print(f"Headphone level decreased to {headphone_offset_db:+.1f} dB")
            elif store_box.contains(mouse):
                store_selected = True
        prev_mouse_down = mouse_down

    sd.default.latency = 'low'

    speaker_audio = make_speaker_audio().astype(np.float32)

    def build_cycle_audio():
        headphone_audio = make_headphone_audio(headphone_offset_db).astype(np.float32)
        headphone_routed = route_to_asio_channels(headphone_audio, 'headphone')
        speaker_routed = route_to_asio_channels(speaker_audio, 'speaker')
        cycle = np.concatenate([headphone_routed, speaker_routed], axis=0)
        return np.concatenate([cycle, cycle], axis=0)

    playback_state = {
        'audio': build_cycle_audio(),
        'pos': 0,
        'rebuild': False,
    }

    def calibration_callback(outdata, frame_count, time_info, status):
        audio = playback_state['audio']
        pos = playback_state['pos']
        end_pos = pos + frame_count
        if end_pos <= audio.shape[0]:
            outdata[:] = audio[pos:end_pos]
            playback_state['pos'] = end_pos
        else:
            first = audio[pos:]
            if playback_state['rebuild']:
                playback_state['audio'] = build_cycle_audio()
                playback_state['rebuild'] = False
                audio = playback_state['audio']
            remaining = frame_count - len(first)
            second = audio[:remaining]
            outdata[:len(first)] = first
            outdata[len(first):] = second
            playback_state['pos'] = remaining

    # Main calibration loop - continuous looping playback with live button polling
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
                draw_calibration_screen("Adjust headphone level during playback; Store ends calibration immediately.", "Playing: HEADPHONE -> LOUDSPEAKER")
                core.wait(0.01)
    finally:
        sd.stop()

    done = visual.TextStim(
        win,
        text=(
            "Calibration complete.\n\n"
            f"Headphone offset to apply: {headphone_offset_db:+.1f} dB\n\n"
            "Press any key to continue to practice trials."
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


def create_playback_streams(headphones_device, speakers_device, sample_rate):
    """Open persistent playback streams for headphone and speaker output."""
    return None, None


def create_experiment_playback_controller(device, sample_rate, channels=4):
    """Create a persistent output stream and shared state for non-blocking trial playback."""
    playback_state = {
        'audio': np.zeros((1, channels), dtype=np.float32),
        'pos': 0,
    }
    state_lock = threading.Lock()

    def callback(outdata, frame_count, time_info, status):
        with state_lock:
            audio = playback_state['audio']
            pos = playback_state['pos']
            end_pos = pos + frame_count

            if pos >= audio.shape[0]:
                outdata[:] = 0
                playback_state['pos'] = end_pos
                return

            if end_pos <= audio.shape[0]:
                outdata[:] = audio[pos:end_pos]
                playback_state['pos'] = end_pos
            else:
                first = audio[pos:]
                outdata[:len(first)] = first
                outdata[len(first):] = 0
                playback_state['pos'] = audio.shape[0]

    stream = sd.OutputStream(
        samplerate=sample_rate,
        device=device,
        channels=channels,
        dtype='float32',
        callback=callback,
        latency='low',
    )
    return stream, playback_state, state_lock


def set_experiment_audio(playback_state, state_lock, audio):
    """Swap in the next trial audio buffer for the persistent playback stream."""
    with state_lock:
        playback_state['audio'] = np.asarray(audio, dtype=np.float32)
        playback_state['pos'] = 0

#load headphone stimuli from /localised_stimuli
base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
headphone_dir = os.path.join(base_dir, 'in_situ_stimuli')
speaker_dir = os.path.join(base_dir, 'loudspeaker_stimuli')
_audio_exts = ('*.wav', '*.flac', '*.mp3', '*.aiff', '*.ogg') 


def _list_audio(folder):
    """Recursively find all audio files in a folder"""
    files = []
    for e in _audio_exts:
        files.extend(glob.glob(os.path.join(folder, '**', e), recursive=True))
    return sorted(files)

# Load stimuli by type (environment, ISTS, noise) for both headphone and speaker
stimulus_types = ['environment', 'ISTS', 'noise']

headphone_stimuli = {}
speaker_stimuli = {}

for stim_type in stimulus_types:
    headphone_stimuli[stim_type] = _list_audio(os.path.join(headphone_dir, stim_type))
    speaker_stimuli[stim_type] = _list_audio(os.path.join(speaker_dir, stim_type))
    print(f"Loaded {len(headphone_stimuli[stim_type])} headphone {stim_type} stimuli")
    print(f"Loaded {len(speaker_stimuli[stim_type])} speaker {stim_type} stimuli")
    
    # Check for empty stimulus lists
    if len(headphone_stimuli[stim_type]) == 0:
        print(f"  WARNING: No headphone {stim_type} files found in {os.path.join(headphone_dir, stim_type)}")
    if len(speaker_stimuli[stim_type]) == 0:
        print(f"  WARNING: No speaker {stim_type} files found in {os.path.join(speaker_dir, stim_type)}")

           
#interstimulus interval
ISI = 1.0 #seconds

# Demographics table: one row per participant
demographics = pd.DataFrame(columns=[
    'participant_id',       # participant ID
    'age',                  # participant age
    'gender',               # participant gender
    'ethnicity',            # participant ethnicity
    'hearing_problems',     # any hearing problems/conditions
    'musician',             # plays instrument or considers self musician
    'musical_experience',   # description of musical experience (if applicable)
    'timestamp'             # when demographics were collected
])

# Trial results table: one row per trial (no demographics)
results = pd.DataFrame(columns=[
    'trial_number',         # trial number (includes practice)
    'trial_type',           # 'practice' or 'experimental'
    'block',                # block number (None for practice)
    'presentation_type',    # 'speaker', 'headphone', or 'eq_headphone'
    'stimulus',             # stimulus filename or ID
    'stimulus_category',    # environment, ISTS, or noise
    'gain_db',              # gain applied to stimulus in dB
    'response',             # key pressed by participant
    'rt',                   # response time in seconds
    'accuracy',             # 1 = correct, 0 = incorrect
    'timestamp'             # trial timestamp
])

# Ensure results directory exists
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Global trial counter
trial_counter = 0

def save_demographics():
    """Save demographics to a separate CSV file."""
    pid = globals().get('participant_id', 'unknown')
    
    demographics.loc[0] = {
        'participant_id': pid,
        'age': globals().get('participant_age', ''),
        'gender': globals().get('participant_gender', ''),
        'ethnicity': globals().get('participant_ethnicity', ''),
        'hearing_problems': globals().get('participant_hearing_problems', ''),
        'musician': globals().get('participant_musician', ''),
        'musical_experience': globals().get('participant_musical_experience', ''),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    fname = os.path.join(results_dir, f"{pid}_demographics.csv")
    try:
        demographics.to_csv(fname, index=False)
        print(f"Demographics saved to {fname}")
    except Exception as e:
        print(f"Warning: failed to save demographics: {e}")


def fetch_individual_eq(participant_id):
    """Find the participant-specific headphone response file exported by DgSonicFocus."""
    eq_file = Path(r"C:\Users\Tim\Documents\DgSonicFocus") / str(participant_id) / 'headphone1_response.txt'
    if eq_file.exists():
        return eq_file

    print(f"Warning: Individual EQ file not found for participant {participant_id}: {eq_file}")
    return None


def _design_peaking_sos(f0_hz, gain_db, q, sample_rate):
    nyquist = sample_rate / 2.0
    if f0_hz <= 0 or f0_hz >= nyquist:
        return None

    amplitude = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (f0_hz / sample_rate)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / amplitude

    if a0 == 0:
        return None

    return np.array([
        b0 / a0,
        b1 / a0,
        b2 / a0,
        1.0,
        a1 / a0,
        a2 / a0,
    ], dtype=np.float64)


def _load_individual_eq(eq_file):
    freqs = []
    gains_db = []

    with Path(eq_file).open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.replace(',', ' ').split()
            try:
                values = [float(value) for value in parts]
            except ValueError:
                continue

            if len(values) < 2:
                continue

            freqs.append(values[0])
            gains_db.append(values[1])

    if not freqs:
        raise ValueError(f"No valid EQ rows found in: {eq_file}")

    freqs = np.asarray(freqs, dtype=np.float64)
    gains_db = np.asarray(gains_db, dtype=np.float64)
    order = np.argsort(freqs)
    return freqs[order], gains_db[order]


def apply_individual_eq(audio, eq_file, sample_rate):
    """Apply participant-specific headphone EQ to audio using a Q=5 peaking filter bank."""
    if eq_file is None:
        return audio

    audio = np.asarray(audio, dtype=np.float32)
    freqs_hz, gains_db = _load_individual_eq(eq_file)

    sos_rows = []
    for f0_hz, gain_db in zip(freqs_hz, gains_db):
        if abs(float(gain_db)) < 1e-9:
            continue
        row = _design_peaking_sos(float(f0_hz), float(gain_db), q=5.0, sample_rate=sample_rate)
        if row is not None:
            sos_rows.append(row)

    if not sos_rows:
        return audio

    sos = np.vstack(sos_rows)
    if audio.ndim == 1:
        processed = signal.sosfilt(sos, audio.astype(np.float64))
        processed = processed.astype(np.float32)
    else:
        channels = []
        for channel_index in range(audio.shape[1]):
            channel_audio = signal.sosfilt(sos, audio[:, channel_index].astype(np.float64))
            channels.append(channel_audio.astype(np.float32))
        processed = np.column_stack(channels)

    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak > 1.0:
        processed = processed / peak

    return processed


def append_result(presentation_type, stimulus, response, rt, accuracy, stimulus_category=None, gain_db=None, trial_type='experimental', block=None):
    """Append one trial's data to the results table and persist to CSV immediately."""
    global trial_counter
    trial_counter += 1
    
    results.loc[len(results)] = {
        'trial_number': trial_counter,
        'trial_type': trial_type,
        'block': block,
        'presentation_type': presentation_type,
        'stimulus': stimulus,
        'stimulus_category': stimulus_category,
        'gain_db': gain_db,
        'response': response,
        'rt': rt,
        'accuracy': accuracy,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }  
    
    # Persist after each trial
    pid = globals().get('participant_id', 'unknown')
    fname = os.path.join(results_dir, f"{pid}_trials.csv")
    try:
        results.to_csv(fname, index=False)
    except Exception as e:
        print(f"Warning: failed to save trial results: {e}")

#launch psychopy window
win = visual.Window(
    size=(1024, 768),
    units='pix',
    fullscr=False,
    color=(0, 0, 0),
    allowStencil=False
)

# hide mouse cursor
win.mouseVisible = False


#ask for participant ID (keyboard input)
participant_id_prompt = visual.TextStim(win, text="Please enter your participant ID and press ENTER:", color='white')
participant_id_prompt.draw()
win.flip()

# Get participant ID input
participant_id = ""
keys = []
while True:
    keys = event.getKeys()
    for key in keys:
        if key == 'return':  # Enter key pressed
            if len(participant_id) > 0:
                break
        elif key == 'backspace':
            participant_id = participant_id[:-1]
        elif key == 'escape':
            win.close()
            core.quit()
        elif len(key) == 1 and key.isalnum():  # alphanumeric characters
            participant_id += key
        
        # Update display with current input
        input_text = f"Please enter your participant ID and press ENTER:\n\n{participant_id}"
        participant_id_prompt.setText(input_text)
        participant_id_prompt.draw()
        win.flip()
    
    if keys and 'return' in keys and len(participant_id) > 0:
        break

print(f"Participant ID: {participant_id}")  # Debug print  

# Function to collect text input from keyboard
def get_text_input(prompt_text, allow_empty=False):
    """Display a prompt and collect keyboard text input."""
    input_str = ""
    prompt = visual.TextStim(win, text=prompt_text, color='white', height=25, wrapWidth=900)
    
    while True:
        # Display prompt with current input
        display_text = f"{prompt_text}\n\n{input_str}"
        prompt.setText(display_text)
        prompt.draw()
        win.flip()
        
        keys = event.getKeys()
        for key in keys:
            if key == 'return':  # Enter key pressed
                if len(input_str) > 0 or allow_empty:
                    return input_str
            elif key == 'backspace':
                input_str = input_str[:-1]
            elif key == 'escape':
                win.close()
                core.quit()
            elif key == 'space':
                input_str += ' '
            elif len(key) == 1:  # Single character (letter, number, punctuation)
                input_str += key

# Collect demographics
print("\nCollecting demographics...")

# Age
participant_age = get_text_input("Please enter your age and press ENTER:")
print(f"Age: {participant_age}")

# Gender
participant_gender = get_text_input("Please enter your gender and press ENTER:")
print(f"Gender: {participant_gender}")

# Ethnicity
participant_ethnicity = get_text_input("Please enter your ethnicity and press ENTER:")
print(f"Ethnicity: {participant_ethnicity}")

# Hearing problems
participant_hearing_problems = get_text_input(
    "Do you have any hearing problems or conditions?\n(Please describe, or type 'no' if none)\n\nPress ENTER when done:"
)
print(f"Hearing problems: {participant_hearing_problems}")

# Musician status
participant_musician = get_text_input(
    "Do you play a musical instrument or consider yourself a musician?\n(yes/no)\n\nPress ENTER when done:"
)
print(f"Musician: {participant_musician}")

# Musical experience (if applicable)
if participant_musician.lower().strip() in ['yes', 'y']:
    participant_musical_experience = get_text_input(
        "Please give a brief description of your musical experience:\n\nPress ENTER when done:"
    )
else:
    participant_musical_experience = "N/A"
print(f"Musical experience: {participant_musical_experience}")

# Save demographics to file
save_demographics()
print("Demographics collection complete.\n")

individual_eq_file = fetch_individual_eq(participant_id)


##trial structure:
# Show fixation cross and wait for ISI
fixation = visual.TextStim(win, text='+', color='white', height=50)

# Audio device indices (adjust these)
ASIO_AGGREGATE_DEVICE = 16  # ASIO4ALL v2 aggregate: 4 output channels
ASIO_SPEAKER_MAPPING = [1, 2]
ASIO_HEADPHONE_MAPPING = [3, 4]
headphones_device = ASIO_AGGREGATE_DEVICE
speakers_device = ASIO_AGGREGATE_DEVICE
sample_rate = 48000  # Default sample rate for audio playback

#repeat for x trials. each new trial should be on a new row in the results table
# Run trials in 3 blocks with breaks
practice_trials = 5

# Generate balanced trial list
number_of_trials = 27 # keep this at 99 for full experiment. multiple of 9
number_of_blocks = 3
trials_per_block_count = number_of_trials // number_of_blocks

if number_of_trials % 9 != 0:
    raise ValueError("number_of_trials must be divisible by 9 (3 outputs x 3 stimulus types)")
if number_of_trials % number_of_blocks != 0:
    raise ValueError("number_of_trials must be divisible by number_of_blocks")


def max_same_output_run(trials):
    """Return longest consecutive run of same output code in a trial list."""
    if not trials:
        return 0
    longest = 1
    current = 1
    previous_output = trials[0][0]
    for output_code, _ in trials[1:]:
        if output_code == previous_output:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            previous_output = output_code
    return longest


def make_balanced_trial_list(n_trials, max_run=2):
    """Create shuffled [output, stim_type] trials with exact 1/3 output balance."""
    trials_per_combination = n_trials // 9
    base_trials = []
    for output in [0, 1, 2]:
        for stim_type in [0, 1, 2]:
            base_trials.extend([[output, stim_type]] * trials_per_combination)

    # Shuffle repeatedly until the output sequence is balanced and not overly clumped.
    for _ in range(2000):
        candidate = base_trials.copy()
        random.shuffle(candidate)
        if max_same_output_run(candidate) <= max_run:
            return candidate

    # Fallback: still balanced, just without the run-length constraint.
    random.shuffle(base_trials)
    return base_trials


trial_list = make_balanced_trial_list(number_of_trials, max_run=2)

# Map numeric codes to string labels
output_map = {0: 'speaker', 1: 'headphone', 2: 'headphone'}
stim_type_map = {0: 'noise', 1: 'ISTS', 2: 'environment'}


def is_headphone_like(playback_type):
    """Return True for headphone-style presentation methods."""
    return playback_type == 'headphone'

print(f"\nTrial configuration:")
print(f"  Total trials: {number_of_trials}")
print(f"  Blocks: {number_of_blocks}")
print(f"  Trials per block: {trials_per_block_count}")
print(f"  Headphone trials: {sum(1 for t in trial_list if t[0] == 1)}")
print(f"  Disabled eq/headphone trials (treated as headphone): {sum(1 for t in trial_list if t[0] == 2)}")
print(f"  Speaker trials: {sum(1 for t in trial_list if t[0] == 0)}")
for stim_code, stim_name in stim_type_map.items():
    print(f"  {stim_name.capitalize()} trials: {sum(1 for t in trial_list if t[1] == stim_code)}")
output_preview = [output_map[t[0]] for t in trial_list[:12]]
print(f"  First 12 trial outputs (shuffled): {output_preview}")
print()

# --- Practice trials ---
# Load image once before trials
img_path = os.path.join(base_dir, 'resources', 'headphonevsloudspeak_info_graphic.png')

# Create info_image once outside the loop for efficiency
if os.path.exists(img_path):
    info_image = visual.ImageStim(
        win,
        image=img_path,
        pos=(0, -150),  # Changed from -280 to -150 (higher on screen)
        size=(400, 200)
    )
else:
    info_image = None
    print("WARNING: Image not found at", img_path)

response_feedback = visual.TextStim(
    win,
    text="",
    color='white',
    height=28,
    pos=(0, 150),
    wrapWidth=1000
)

# Loudness calibration before practice trials
headphone_level_offset_db, speaker_level_offset_db = run_loudness_calibration(
    win,
    headphones_device=headphones_device,
    speakers_device=speakers_device,
    sample_rate=sample_rate
)

# Display main task instructions after calibration
instructions_text = """You will hear sounds either through headphones or loudspeakers.

Your task is to identify whether the sound is played through headphones or loudspeakers.

Press the UP ARROW key for loudspeakers and the DOWN ARROW key for headphones.

Try to respond as quickly and accurately as possible after you see the image of headphones and loudspeakers. You can't respond until you see them

Please keep your head as still as possible and look at the fixation cross throughout

Press any key to begin."""

instructions = visual.TextStim(
    win,
    text=instructions_text,
    color='white',
    height=30,
    wrapWidth=1100
)
instructions.draw()
win.flip()
event.waitKeys()
win.flip()

for p in range(practice_trials):
        # Show fixation cross with ISI
        fixation.draw()
        win.flip()
        core.wait(ISI)

        # Randomly choose playback and stimulus category for practice
        playback_type = random.choice(['headphone', 'speaker'])
        stim_category = random.choice(stimulus_types)

        if is_headphone_like(playback_type):
            stimulus = random.choice(headphone_stimuli[stim_category])
            device = headphones_device
            mapping = ASIO_HEADPHONE_MAPPING
        else:
            stimulus = random.choice(speaker_stimuli[stim_category])
            device = speakers_device
            mapping = ASIO_SPEAKER_MAPPING

        response = None
        rt = 0
        try:
            audio_data, fs = sf.read(stimulus)
            samples_5s = int(5 * fs)
            audio_5s = audio_data[:samples_5s]
            gain_db = random.uniform(0, 0) # random gain can be adjusted currently zero
            if is_headphone_like(playback_type):
                gain_db += headphone_level_offset_db
            else:
                gain_db += speaker_level_offset_db
            gain_linear = 10 ** (gain_db / 20)
            audio_5s = audio_5s * gain_linear

            if audio_5s.ndim > 1 and playback_type == 'speaker':
                audio_5s = audio_5s.mean(axis=1)
            audio_5s = apply_device_specific_filter(audio_5s, fs, device_type='headphone' if is_headphone_like(playback_type) else 'speaker', order=4)

            if playback_type == 'speaker':
                audio_5s = collapse_to_left_channel(audio_5s, preserve_total_rms=True)

            audio_5s = ensure_stereo(audio_5s)
            audio_5s = apply_fade(audio_5s, fs, fade_ms=20)
            audio_5s = append_silence_tail(audio_5s, fs, tail_ms=20)

            print(f"Practice {p+1}/{practice_trials}: {playback_type} via device {device} ({stim_category})")

            fixation.draw()
            win.flip()

            sd.play(audio_5s, samplerate=fs, device=device, mapping=mapping)
            start_time = time.time()
            audio_duration = len(audio_5s) / fs

            image_shown = False
            response = None
            response_message = ""

            while True:
                elapsed_time = time.time() - start_time

                if not image_shown and elapsed_time >= 3.0:
                    image_shown = True

                if image_shown and response is None:
                    keys = event.getKeys(keyList=['up', 'down', 'escape'], timeStamped=False)
                    if keys:
                        if 'escape' in keys:
                            sd.stop()
                            win.close()
                            core.quit()
                        if 'up' in keys:
                            response = 'up'
                            rt = elapsed_time
                            response_message = "response recorded - loudspeaker"
                        elif 'down' in keys:
                            response = 'down'
                            rt = elapsed_time
                            response_message = "response recorded - headphone"
                elif not image_shown:
                    event.getKeys()

                fixation.draw()
                if image_shown and info_image:
                    info_image.draw()
                if image_shown and response_message:
                    response_feedback.setText(response_message)
                    response_feedback.draw()
                win.flip()

                if elapsed_time >= audio_duration:
                    break

                core.wait(0.01)

            sd.wait()

            while response is None:
                keys = event.getKeys(keyList=['up', 'down', 'escape'], timeStamped=False)
                if keys:
                    if 'escape' in keys:
                        sd.stop()
                        win.close()
                        core.quit()
                    if 'up' in keys:
                        response = 'up'
                        rt = time.time() - start_time
                        response_message = "response recorded - loudspeaker"
                    elif 'down' in keys:
                        response = 'down'
                        rt = time.time() - start_time
                        response_message = "response recorded - headphone"

                fixation.draw()
                if info_image:
                    info_image.draw()
                response_feedback.setText(response_message if response_message else "please respond: up=loudspeaker, down=headphone")
                response_feedback.draw()
                win.flip()
                core.wait(0.01)

        except Exception as e:
            print(f"Error playing practice stimulus {stimulus}: {e}")
            sd.stop()
            continue

# --- End practice trials ---

# --- Main experimental trials ---
trial_index = 0

# Create a persistent output stream for trial playback (4 channels for headphone + speaker)
trial_playback_state = {
    'audio': np.zeros((1, 4), dtype=np.float32),
    'pos': 0,
    'audio_duration_samples': 0,
}
trial_state_lock = threading.Lock()

def trial_playback_callback(outdata, frame_count, time_info, status):
    """Callback for continuous trial playback stream."""
    with trial_state_lock:
        audio = trial_playback_state['audio']
        pos = trial_playback_state['pos']
        audio_duration = trial_playback_state['audio_duration_samples']
        end_pos = pos + frame_count

        # If we've reached the end of the audio, output silence
        if pos >= audio_duration:
            outdata[:] = 0
            trial_playback_state['pos'] = end_pos
            return

        # If the requested frame count fits within remaining audio
        if end_pos <= audio_duration:
            outdata[:] = audio[pos:end_pos]
            trial_playback_state['pos'] = end_pos
        else:
            # Output remaining audio and pad with silence
            first = audio[pos:audio_duration]
            outdata[:len(first)] = first
            outdata[len(first):] = 0
            trial_playback_state['pos'] = audio_duration

# Open the persistent trial playback stream
trial_stream = sd.OutputStream(
    samplerate=sample_rate,
    device=ASIO_AGGREGATE_DEVICE,
    channels=4,
    dtype='float32',
    callback=trial_playback_callback,
    latency='low',
)
trial_stream.start()

try:
    for block in range(number_of_blocks):
        for i in range(trials_per_block_count):
            # Show fixation cross with ISI
            fixation.draw()
            win.flip()
            core.wait(ISI)
            
            # Get trial configuration: [output, stim_type]
            output_code, stim_type_code = trial_list[trial_index]
            playback_type = output_map[output_code]
            stim_category = stim_type_map[stim_type_code]
            
            if is_headphone_like(playback_type):
                stimulus = random.choice(headphone_stimuli[stim_category])
            else:
                stimulus = random.choice(speaker_stimuli[stim_category])
            
            response = None  # Initialize response variable
            rt = 0
            try:
                audio_data, fs = sf.read(stimulus)
                # Play only first 5 seconds
                samples_5s = int(5 * fs)
                audio_5s = audio_data[:samples_5s]
                # Apply random gain between -5 and +5 dB
                gain_db = random.uniform(-2 , 2)
                if is_headphone_like(playback_type):
                    gain_db += headphone_level_offset_db
                else:
                    gain_db += speaker_level_offset_db
                gain_linear = 10 ** (gain_db / 20)
                audio_5s = audio_5s * gain_linear

                if audio_5s.ndim > 1 and playback_type == 'speaker':
                    audio_5s = audio_5s.mean(axis=1)
                audio_5s = apply_device_specific_filter(audio_5s, fs, device_type='headphone' if is_headphone_like(playback_type) else 'speaker', order=4)

                if playback_type == 'speaker':
                    audio_5s = collapse_to_left_channel(audio_5s, preserve_total_rms=True)

                audio_5s = ensure_stereo(audio_5s)
                audio_5s = apply_fade(audio_5s, fs, fade_ms=20)
                audio_5s = append_silence_tail(audio_5s, fs, tail_ms=20)
                
                # Route audio to appropriate ASIO channels
                routed_audio = route_to_asio_channels(audio_5s, playback_type)

                # Debug: log routing
                print(f"Trial {trial_index+1}/{number_of_trials}: {playback_type} ({stim_category})")
                
                # Set up playback state with the new audio
                with trial_state_lock:
                    trial_playback_state['audio'] = routed_audio.astype(np.float32)
                    trial_playback_state['pos'] = 0
                    trial_playback_state['audio_duration_samples'] = routed_audio.shape[0]
                
                # Show only fixation cross initially
                fixation.draw()
                win.flip()
                
                start_time = time.time()
                audio_duration = len(audio_5s) / fs
                
                # Wait 3 seconds, then show image below fixation cross
                image_shown = False
                response = None
                response_message = ""
                
                while True:
                    elapsed_time = time.time() - start_time
                    
                    # Check if 3 seconds has passed and image hasn't been shown yet
                    if not image_shown and elapsed_time >= 3.0:
                        image_shown = True
                    
                    # Only check for responses AFTER image is shown (after 3 seconds)
                    if image_shown and response is None:
                        keys = event.getKeys(keyList=['up', 'down', 'escape'], timeStamped=False)
                        if keys:
                            if 'escape' in keys:
                                trial_stream.stop()
                                trial_stream.close()
                                win.close()
                                core.quit()
                            if 'up' in keys:
                                response = 'up'
                                rt = elapsed_time
                                response_message = "response recorded - loudspeaker"
                            elif 'down' in keys:
                                response = 'down'
                                rt = elapsed_time
                                response_message = "response recorded - headphone"
                    elif not image_shown:
                        # Clear any key presses before 3 seconds (ignore them)
                        event.getKeys()

                    fixation.draw()
                    if image_shown and info_image:
                        info_image.draw()
                    if image_shown and response_message:
                        response_feedback.setText(response_message)
                        response_feedback.draw()
                    win.flip()

                    # Playback always runs to completion regardless of response timing
                    if elapsed_time >= audio_duration:
                        break
                    
                    core.wait(0.01)  # Small delay to prevent CPU overload

                # Require a response before advancing to next trial
                while response is None:
                    keys = event.getKeys(keyList=['up', 'down', 'escape'], timeStamped=False)
                    if keys:
                        if 'escape' in keys:
                            trial_stream.stop()
                            trial_stream.close()
                            win.close()
                            core.quit()
                        if 'up' in keys:
                            response = 'up'
                            rt = time.time() - start_time
                            response_message = "response recorded - loudspeaker"
                        elif 'down' in keys:
                            response = 'down'
                            rt = time.time() - start_time
                            response_message = "response recorded - headphone"

                    fixation.draw()
                    if info_image:
                        info_image.draw()
                    response_feedback.setText(response_message if response_message else "please respond: up=loudspeaker, down=headphone")
                    response_feedback.draw()
                    win.flip()
                    core.wait(0.01)

            except Exception as e:
                print(f"Error playing stimulus {stimulus} ({playback_type}): {e}")
                trial_index += 1
                continue

            if (response == 'up' and playback_type == 'speaker') or (response == 'down' and is_headphone_like(playback_type)):
                accuracy = 1
            else:
                accuracy = 0

            append_result(playback_type, stimulus, response, rt, accuracy,
                         stimulus_category=stim_category, gain_db=gain_db, trial_type='experimental', block=block+1)
            trial_index += 1

        if block < (number_of_blocks - 1):
            break_msg = visual.TextStim(
                win,
                text=f"You're {block + 1}/{number_of_blocks} of the way through.\n\nTake a break.\n\nPress any key to continue.",
                color='white',
                height=30
            )
            break_msg.draw()
            win.flip()
            event.waitKeys()

finally:
    # Close trial stream
    if trial_stream:
        trial_stream.stop()
        trial_stream.close()

# --- End of experiment ---
print("\nExperiment complete!")

# Display thank you message
thank_you_text = visual.TextStim(
    win,
    text="Thank you for participating!\n\nYour results have been stored.\n\nPlease find Tim in the corridor.",
    color='white',
    height=35,
    wrapWidth=900
)
thank_you_text.draw()
win.flip()

# Wait for any key press before closing
event.waitKeys()

# Close window and quit
win.close()
core.quit()

