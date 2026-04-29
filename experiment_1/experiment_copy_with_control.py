#this is the same as pilot experiment but with the perceptual loudness calibration i'm working on

from psychopy import visual, event, core
import pandas as pd
import time
import numpy as np
import random
import sounddevice as sd
import soundfile as sf
import os, glob
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


def apply_device_specific_filter(audio, sample_rate, device_type='headphone', order=4):
    """
    Apply device-specific EQ to match frequency response between headphones and speakers.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        device_type: 'headphone' or 'speaker'
        order: Filter order
    
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
        # High-pass: 60 Hz (less aggressive)
        # Low-pass: 11000 Hz (more treble)
        hp_freq, lp_freq = 60, 11000
    
    # High-pass filter
    sos_hp = signal.butter(order, hp_freq, btype='high', fs=sample_rate, output='sos')
    audio = signal.sosfilt(sos_hp, audio, axis=0)
    
    # Low-pass filter
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


def collapse_to_left_channel(audio):
    """
    Collapse audio to left channel only for loudspeaker playback.
    
    Converts stereo/mono to stereo with left = audio, right = 0 (silence).
    
    Args:
        audio: Audio array (mono or stereo)
    
    Returns:
        Stereo audio array with right channel silenced
    """
    if audio.ndim == 1:
        # Mono: convert to stereo with left = audio, right = silence
        stereo = np.zeros((len(audio), 2), dtype=audio.dtype)
        stereo[:, 0] = audio
        return stereo
    elif audio.shape[1] >= 2:
        # Stereo or multi-channel: keep left (channel 0), silence right
        stereo = np.zeros_like(audio[:, :2])
        stereo[:, 0] = audio[:, 0]
        return stereo
    else:
        # Single channel stereo-like array: convert to stereo
        stereo = np.zeros((audio.shape[0], 2), dtype=audio.dtype)
        stereo[:, 0] = audio[:, 0]
        return stereo

def run_loudness_calibration(win, headphones_device, speakers_device, sample_rate=48000):
    """Calibrate headphone level against a fixed speaker reference using on-screen buttons."""
    if headphones_device == speakers_device:
        raise ValueError("Headphone and speaker device indices are the same. Set different device indices before calibration.")

    rng = np.random.default_rng(42)
    duration_s = 1.0
    reference_audio = rng.standard_normal(int(sample_rate * duration_s)).astype(np.float32)
    reference_audio = reference_audio / np.max(np.abs(reference_audio)) * 0.5

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
        audio = collapse_to_left_channel(audio)
        audio = apply_fade(audio, sample_rate, fade_ms=20)
        return audio

    def make_headphone_audio(current_hp_offset_db):
        audio = apply_gain_db(reference_headphone_eq, current_hp_offset_db)
        audio = apply_fade(audio, sample_rate, fade_ms=20)
        return audio

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

    phase_lock = threading.Lock()
    current_phase = "LOUDSPEAKER"
    stop_event = threading.Event()
    playback_error = None

    speaker_audio = make_speaker_audio().astype(np.float32)

    speaker_stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        dtype='float32',
        latency='low',
        device=speakers_device,
    )
    headphone_stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        dtype='float32',
        latency='low',
        device=headphones_device,
    )

    def playback_loop():
        nonlocal current_phase, headphone_offset_db, playback_error
        try:
            while not stop_event.is_set():
                with phase_lock:
                    current_phase = "LOUDSPEAKER"
                speaker_stream.write(speaker_audio)
                if stop_event.is_set():
                    break

                with phase_lock:
                    current_phase = "HEADPHONE"
                    hp_offset = headphone_offset_db
                headphone_audio = make_headphone_audio(hp_offset).astype(np.float32)
                headphone_stream.write(headphone_audio)
        except Exception as e:
            playback_error = e
            stop_event.set()

    try:
        speaker_stream.start()
        headphone_stream.start()

        playback_thread = threading.Thread(target=playback_loop, daemon=True)
        playback_thread.start()

        while not store_selected:
            if playback_error is not None:
                raise RuntimeError(f"Calibration playback failed: {playback_error}")

            with phase_lock:
                phase_label = f"Playing: {current_phase}"
            draw_calibration_screen(
                "Click + or - while alternating playback runs, or Store to finish.",
                phase_label,
            )
            maybe_handle_click()
            core.wait(0.01)

        stop_event.set()
        playback_thread.join(timeout=2.5)
    finally:
        try:
            speaker_stream.stop()
            speaker_stream.close()
        except Exception:
            pass
        try:
            headphone_stream.stop()
            headphone_stream.close()
        except Exception:
            pass

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
    speaker_stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        dtype='float32',
        latency='low',
        device=speakers_device,
    )
    headphone_stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        dtype='float32',
        latency='low',
        device=headphones_device,
    )
    speaker_stream.start()
    headphone_stream.start()
    return speaker_stream, headphone_stream

base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
headphone_root_candidates = [
    os.path.join(base_dir, 'normalised_localised'),
    os.path.join(base_dir, 'localised_stimuli_B20'),
    os.path.join(base_dir, 'localised_stimuli_b20'),
    os.path.join(base_dir, 'localised_stimuli'),
]
speaker_root_candidates = [
    os.path.join(base_dir, 'normalised_loudspeaker'),
    os.path.join(base_dir, 'loudspeaker_stimuli'),
]

_audio_exts = ('*.wav', '*.flac', '*.mp3', '*.aiff', '*.ogg')


def _list_audio(folder):
    """Recursively find all audio files in a folder"""
    files = []
    for e in _audio_exts:
        files.extend(glob.glob(os.path.join(folder, '**', e), recursive=True))
    return sorted(files)

# Load stimuli by type (environment, ISTS, noise) for both headphone and speaker
stimulus_types = ['environment', 'ISTS', 'noise']


def load_stimulus_sets(root_candidates, label):
    for root in root_candidates:
        loaded = {}
        for stim_type in stimulus_types:
            loaded[stim_type] = _list_audio(os.path.join(root, stim_type))
        if all(loaded[stim_type] for stim_type in stimulus_types):
            print(f"Using {label} stimuli from {root}")
            for stim_type in stimulus_types:
                print(f"Loaded {len(loaded[stim_type])} {label} {stim_type} stimuli")
            return root, loaded

    raise FileNotFoundError(
        f"No complete {label} stimulus set found in any of: {root_candidates}"
    )


headphone_dir, headphone_stimuli = load_stimulus_sets(headphone_root_candidates, 'headphone')
speaker_dir, speaker_stimuli = load_stimulus_sets(speaker_root_candidates, 'speaker')

           
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
    'presentation_type',    # 'headphone' or 'speaker'
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
    fullscr=True,
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


##trial structure:
# Show fixation cross and wait for ISI
fixation = visual.TextStim(win, text='+', color='white', height=50)

# Audio device indices (adjust these)
headphones_device = 25  # Device index for headphones
speakers_device = 5  # Device index for speakers


#repeat for x trials. each new trial should be on a new row in the results table
# Run trials in 3 blocks with breaks
practice_trials = 5

# Generate balanced trial list
number_of_trials = 24 #keep this at 96 for full experiment. multiple of 6
number_of_blocks = 3
trials_per_block_count = number_of_trials // number_of_blocks

if number_of_trials % 6 != 0:
    raise ValueError("number_of_trials must be divisible by 6 (2 outputs x 3 stimulus types)")
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
    """Create shuffled [output, stim_type] trials with exact 50/50 output balance."""
    trials_per_combination = n_trials // 6
    base_trials = []
    for output in [0, 1]:
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
output_map = {0: 'speaker', 1: 'headphone'}
stim_type_map = {0: 'noise', 1: 'ISTS', 2: 'environment'}

print(f"\nTrial configuration:")
print(f"  Total trials: {number_of_trials}")
print(f"  Blocks: {number_of_blocks}")
print(f"  Trials per block: {trials_per_block_count}")
print(f"  Headphone trials: {sum(1 for t in trial_list if t[0] == 1)}")
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
    speakers_device=speakers_device
)

speaker_trial_stream, headphone_trial_stream = create_playback_streams(
    headphones_device=headphones_device,
    speakers_device=speakers_device,
    sample_rate=48000,
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

    if playback_type == 'headphone':
        stimulus = random.choice(headphone_stimuli[stim_category])
        device = headphones_device
    else:
        stimulus = random.choice(speaker_stimuli[stim_category])
        device = speakers_device

    response = None
    rt = 0
    try:
        audio_data, fs = sf.read(stimulus)
        samples_5s = int(5 * fs)
        audio_5s = audio_data[:samples_5s]
        gain_db = random.uniform(-2, 2)
        if playback_type == 'headphone':
            gain_db += headphone_level_offset_db
        else:
            gain_db += speaker_level_offset_db
        gain_linear = 10 ** (gain_db / 20)
        audio_5s = audio_5s * gain_linear

        if audio_5s.ndim > 1 and playback_type == 'speaker':
            audio_5s = audio_5s.mean(axis=1)
        audio_5s = apply_device_specific_filter(audio_5s, fs, device_type='headphone' if playback_type == 'headphone' else 'speaker', order=4)

        if playback_type == 'speaker':
            audio_5s = collapse_to_left_channel(audio_5s)
        else:
            audio_5s = ensure_stereo(audio_5s)
        audio_5s = apply_fade(audio_5s, fs, fade_ms=10)
        audio_5s = append_silence_tail(audio_5s, fs, tail_ms=20)

        print(f"Practice {p+1}/{practice_trials}: {playback_type} via device {device} ({stim_category})")

        # Show only fixation cross initially
        fixation.draw()
        win.flip()

        # START AUDIO PLAYBACK (non-blocking) and start timing immediately
        playback_error = None
        target_stream = headphone_trial_stream if playback_type == 'headphone' else speaker_trial_stream
        playback_thread = threading.Thread(
            target=lambda: play_audio_on_stream(audio_5s, target_stream),
            daemon=True,
        )
        playback_thread.start()
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

        playback_thread.join(timeout=2.0)

        # Require a response before advancing to next trial
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
        print(f"Error playing practice stimulus {stimulus} ({playback_type}, device {device}): {e}")
        sd.stop()  # Ensure audio is stopped on error
        continue

    if (response == 'up' and playback_type == 'speaker') or (response == 'down' and playback_type == 'headphone'):
        accuracy = 1
    else:
        accuracy = 0

    # Save practice trial (trial_type='practice', block=None)
    append_result(playback_type, stimulus, response, rt, accuracy,
                 stimulus_category=stim_category, gain_db=gain_db, trial_type='practice', block=None)

# --- End practice trials ---

# --- Main experimental trials ---
trial_index = 0  # Separate counter for indexing trial_list

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
        
        if playback_type == 'headphone':
            stimulus = random.choice(headphone_stimuli[stim_category])
            device = headphones_device
        else:
            stimulus = random.choice(speaker_stimuli[stim_category])
            device = speakers_device
        
        response = None  # Initialize response variable
        rt = 0
        try:
            audio_data, fs = sf.read(stimulus)
            # Play only first 5 seconds
            samples_5s = int(5 * fs)
            audio_5s = audio_data[:samples_5s]
            # Apply random gain between -5 and +5 dB
            gain_db = random.uniform(-2 , 2)
            if playback_type == 'headphone':
                gain_db += headphone_level_offset_db
            else:
                gain_db += speaker_level_offset_db
            gain_linear = 10 ** (gain_db / 20)
            audio_5s = audio_5s * gain_linear

            if audio_5s.ndim > 1 and playback_type == 'speaker':
                audio_5s = audio_5s.mean(axis=1)
            audio_5s = apply_device_specific_filter(audio_5s, fs, device_type='headphone' if playback_type == 'headphone' else 'speaker', order=4)
            if playback_type == 'speaker':
                audio_5s = collapse_to_left_channel(audio_5s)
            else:
                audio_5s = ensure_stereo(audio_5s)
            audio_5s = apply_fade(audio_5s, fs, fade_ms=10)
            audio_5s = append_silence_tail(audio_5s, fs, tail_ms=20)

            # Explicitly set output device (leave input as default)
            # Debug: log routing
            print(f"Trial {trial_index+1}/{number_of_trials}: {playback_type} via device {device} ({stim_category})")
            
            # Show only fixation cross initially
            fixation.draw()
            win.flip()
            
            # Start audio playback (non-blocking) and start timing immediately
            playback_error = None
            target_stream = headphone_trial_stream if playback_type == 'headphone' else speaker_trial_stream
            playback_thread = threading.Thread(
                target=lambda: play_audio_on_stream(audio_5s, target_stream),
                daemon=True,
            )
            playback_thread.start()
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

            playback_thread.join(timeout=2.0)

            # Require a response before advancing to next trial
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
            print(f"Error playing stimulus {stimulus} ({playback_type}, device {device}): {e}")
            sd.stop()  # Ensure audio is stopped on error
            trial_index += 1  # Still increment to avoid getting stuck
            continue  # Skip this trial if error occurs

        if (response == 'up' and playback_type == 'speaker') or (response == 'down' and playback_type == 'headphone'):
            accuracy = 1
        else:
            accuracy = 0

        # Save experimental trial (trial_type='experimental', block=block number)
        append_result(playback_type, stimulus, response, rt, accuracy,
                     stimulus_category=stim_category, gain_db=gain_db, trial_type='experimental', block=block+1)
        trial_index += 1  # Increment after successful trial

    # Display break message after each block (except the last)
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

# Close playback streams
try:
    speaker_trial_stream.stop()
    speaker_trial_stream.close()
except Exception:
    pass

try:
    headphone_trial_stream.stop()
    headphone_trial_stream.close()
except Exception:
    pass

# Close window and quit
win.close()
core.quit()

