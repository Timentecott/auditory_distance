from psychopy import visual, event, core
from psychopy.hardware import keyboard
import os
os.environ["SD_ENABLE_ASIO"] = "1" #this line is important as it allows revelation of asio devices
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
import random
from threading import Lock

# Experiment parameters
NUMBER_OF_TRIALS = 64  # Must be divisible by 8 (4 locations x 2 validity types)
FIXATION_DURATION = 0.5  # seconds
SOUND_CUE_DURATION = 0.1  # seconds
CUE_TO_DOT_ISI = 0.1  # seconds
DOT_DURATION = 0.1  # seconds
INTER_TRIAL_INTERVAL = 1.0  # seconds
RESPONSE_TIMEOUT = 3.0  # Maximum time to wait for response in seconds
# Set to the ASIO aggregate output device index that exposes 4 output channels.
# Channels 3-4 are used for headphone playback.
AUDIO_OUTPUT_DEVICE_INDEX = 18

# Screen coordinates for locations
DISTANCE_FROM_CENTER = 200  # pixels
LOCATIONS = {
    'far_left': (-DISTANCE_FROM_CENTER, DISTANCE_FROM_CENTER),
    'far_right': (DISTANCE_FROM_CENTER, DISTANCE_FROM_CENTER),
    'near_left': (-DISTANCE_FROM_CENTER, -DISTANCE_FROM_CENTER),
    'near_right': (DISTANCE_FROM_CENTER, -DISTANCE_FROM_CENTER)
}

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(base_dir, 'audio_stimuli')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Audio file naming prefix
AUDIO_FILE_PREFIX = 'pink_noise_48k_30s_300_8000hz'

# Map location names to audio file suffixes
AUDIO_FILE_MAPPING = {
    'near_left': 'near_left',
    'near_right': 'near_right',
    'far_left': 'far_left',
    'far_right': 'far_right'
}

participant_id = ""
demographics = pd.DataFrame(columns=['participant_id'])

START_KEYS = {'num_5', 'num5', 'kp_5', 'numpad5', '5', 'clear'}

# Accept both arrow keys and common numpad key names across keyboards/backends.
# Maps: 1=near_left, 3=near_right, 7=far_left, 9=far_right
RESPONSE_KEY_MAP = {
    'left': 'near_left',
    'right': 'near_right',
    'up': 'far_left',
    'down': 'far_right',
    'num_1': 'near_left',
    'num1': 'near_left',
    'kp_1': 'near_left',
    'numpad1': 'near_left',
    '1': 'near_left',
    'num_3': 'near_right',
    'num3': 'near_right',
    'kp_3': 'near_right',
    'numpad3': 'near_right',
    '3': 'near_right',
    'num_7': 'far_left',
    'num7': 'far_left',
    'kp_7': 'far_left',
    'numpad7': 'far_left',
    '7': 'far_left',
    'num_9': 'far_right',
    'num9': 'far_right',
    'kp_9': 'far_right',
    'numpad9': 'far_right',
    '9': 'far_right',
}

cue_playback_state = {
    'audio': None,
    'pos': 0,
    'audio_duration_samples': 0
}
cue_state_lock = Lock()

def wait_for_start_key():
    event.clearEvents()
    kb.clearEvents()

    while True:
        key_names = []

        kb_keys = kb.getKeys(waitRelease=False, clear=True)
        key_names.extend([str(k.name).strip().lower() for k in kb_keys])

        event_keys = event.getKeys()
        key_names.extend([str(k).strip().lower() for k in event_keys])

        if key_names:
            print(f"Start screen key(s): {key_names}")

        if 'escape' in key_names:
            win.close()
            core.quit()

        if any((k in START_KEYS) or ('5' in k) for k in key_names):
            return

        core.wait(0.01)


def get_direction_response():
    """Return canonical direction ('left'/'right'/'up'/'down') or 'escape'."""
    key_names = []

    kb_keys = kb.getKeys(waitRelease=False, clear=True)
    key_names.extend([str(k.name).strip().lower() for k in kb_keys])

    event_keys = event.getKeys()
    key_names.extend([str(k).strip().lower() for k in event_keys])

    for key in key_names:
        if key == 'escape':
            return 'escape'
        mapped = RESPONSE_KEY_MAP.get(key)
        if mapped:
            return mapped

    return None


def play_cue(location_name):
    """Play the first 100 ms of a cue using the persistent output stream."""
    cue_audio, cue_sr = cue_sounds[location_name]
    cue_samples = int(round(cue_sr * SOUND_CUE_DURATION))
    cue_audio = cue_audio[:cue_samples]

    with cue_state_lock:
        cue_playback_state['audio'] = np.asarray(cue_audio, dtype=np.float32)
        cue_playback_state['pos'] = 0
        cue_playback_state['audio_duration_samples'] = cue_audio.shape[0]

    while True:
        with cue_state_lock:
            if cue_playback_state['pos'] >= cue_playback_state['audio_duration_samples']:
                break
        core.wait(0.001)


# Create PsychoPy window
win = visual.Window(
    size=[1920, 1080],
    fullscr=True,
    color='gray',
    units='pix'
)
win.mouseVisible = False
kb = keyboard.Keyboard()


def get_text_input(prompt_text):
    input_str = ""
    prompt = visual.TextStim(win, text=prompt_text, color='white', height=30, wrapWidth=1200)
    while True:
        prompt.setText(f"{prompt_text}\n\n{input_str}")
        prompt.draw()
        win.flip()
        for key in event.getKeys():
            if key == 'return':
                if input_str.strip():
                    return input_str.strip()
            elif key == 'backspace':
                input_str = input_str[:-1]
            elif key == 'escape':
                win.close()
                core.quit()
            elif len(key) == 1:
                input_str += key


def save_demographics():
    demographics.loc[0] = {'participant_id': participant_id}
    demographics_file = os.path.join(results_dir, f'{participant_id}_demographics.csv')
    demographics.to_csv(demographics_file, index=False)
    print(f"Demographics saved to {demographics_file}")


# Collect participant ID
participant_id = get_text_input("Please enter your participant ID and press ENTER:")
save_demographics()

# Preload cue sounds once to reduce onset latency during trials.
cue_sounds = {}
for loc in LOCATIONS:
    audio_suffix = AUDIO_FILE_MAPPING[loc]
    cue_path = os.path.join(audio_dir, f'{AUDIO_FILE_PREFIX}_{audio_suffix}.wav')
    if not os.path.exists(cue_path):
        raise FileNotFoundError(f"Missing audio cue file: {cue_path}")
    cue_audio, cue_sr = sf.read(cue_path, dtype='float32')
    cue_sounds[loc] = (cue_audio, cue_sr)


def cue_playback_callback(outdata, frame_count, time_info, status):
    with cue_state_lock:
        audio = cue_playback_state['audio']
        pos = cue_playback_state['pos']
        audio_duration = cue_playback_state['audio_duration_samples']

        if audio is None or pos >= audio_duration:
            outdata[:] = 0
            cue_playback_state['pos'] = pos + frame_count
            return

        end_pos = pos + frame_count
        audio_frame = audio[pos:min(end_pos, audio_duration)]

        if audio_frame.ndim == 1:
            audio_frame = np.column_stack([audio_frame, audio_frame])
        else:
            audio_frame = audio_frame[:, :2]

        routed = np.zeros((frame_count, 4), dtype=np.float32)
        routed[:len(audio_frame), 2:4] = audio_frame
        outdata[:] = routed
        cue_playback_state['pos'] = min(end_pos, audio_duration)

cue_stream = sd.OutputStream(
    samplerate=48000,
    device=AUDIO_OUTPUT_DEVICE_INDEX,
    channels=4,
    dtype='float32',
    callback=cue_playback_callback,
    latency='low',
)
cue_stream.start()

# Create visual stimuli
fixation = visual.ShapeStim(
    win,
    vertices='cross',
    size=30,
    lineColor='white',
    fillColor='white'
)

dot = visual.Circle(
    win,
    radius=20,
    fillColor='white',
    lineColor='white'
)

practice_instructions = visual.TextStim(
    win,
    text="You will hear a sound, then see a dot.\nPress an arrow key on the numpad (4=Near Left, 6=Near Right, 8=Far Left, 2=Far Right) to indicate where the dot is as quickly as possible.\nOnly use one finger for the duration of the experiment\n\nPress 5 on the numpad to have a practice.",
    color='white',
    height=30,
    wrapWidth=1000,
    pos=(0, 180)
)
practice_image_path = os.path.join(base_dir, 'visual stimuli', 'numpad_pic.png')
practice_image = visual.ImageStim(
    win,
    image=practice_image_path,
    pos=(0, -120),
    size=(500, 350)
)
instructions = visual.TextStim(
    win,
    text="practice complete, remember to press the button according to where you see the circle and ignore the sound. please press 5 to start the main experiment, you will not get feedback in the main experiment",
    color='white',
    height=30,
    wrapWidth=1000
)

def determine_validity_condition(sound_loc, dot_loc):
    """
    Determine validity condition based on audio (sound) and visual (dot) locations.
    Categories:
    - Valid: same location
    - Azimuth Invalid: same distance but different side (left/right)
    - Distance Invalid: same side but different distance (near/far)
    - Double Invalid: different distance and different side
    """
    if sound_loc == dot_loc:
        return 'Valid'
    
    # Extract distance and azimuth from location names
    sound_parts = sound_loc.split('_')  # e.g., 'near_left' -> ['near', 'left']
    dot_parts = dot_loc.split('_')      # e.g., 'far_right' -> ['far', 'right']
    
    sound_distance = sound_parts[0]  # 'near' or 'far'
    sound_azimuth = sound_parts[1]   # 'left' or 'right'
    dot_distance = dot_parts[0]
    dot_azimuth = dot_parts[1]
    
    distance_match = sound_distance == dot_distance
    azimuth_match = sound_azimuth == dot_azimuth
    
    if distance_match and not azimuth_match:
        return 'Azimuth Invalid'
    elif azimuth_match and not distance_match:
        return 'Distance Invalid'
    else:  # not distance_match and not azimuth_match
        return 'Double Invalid'


# Display practice instructions
practice_instructions.draw()
practice_image.draw()
win.flip()
wait_for_start_key()

# do 4 practice trials, including feedback for each one
practice_location_names = ['near_left', 'near_right', 'far_left', 'far_right']
practice_trials = []

for loc in practice_location_names:
    # Create one valid practice trial per location
    practice_trials.append({'sound_location': loc, 'dot_location': loc})

random.shuffle(practice_trials)

for practice_trial in practice_trials:
    sound_location = practice_trial['sound_location']
    dot_location = practice_trial['dot_location']
    validity_condition = determine_validity_condition(sound_location, dot_location)

    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)

    play_cue(sound_location)

    win.flip()
    core.wait(CUE_TO_DOT_ISI)

    dot.pos = LOCATIONS[dot_location]
    dot.draw()
    win.flip()
    core.wait(DOT_DURATION)

    win.flip()

    event.clearEvents()
    kb.clearEvents()

    response = None
    while response is None:
        response = get_direction_response()

    if response == 'escape':
        win.close()
        core.quit()

    correct = response == dot_location
    feedback = visual.TextStim(
        win,
        text='Correct!' if correct else 'Incorrect',
        color='green' if correct else 'red',
        height=40
    )
    feedback.draw()
    win.flip()
    core.wait(0.5)

    win.flip()
    core.wait(INTER_TRIAL_INTERVAL)

# display instrunctions for main experiment
instructions.draw()
win.flip()
wait_for_start_key()

# Create trial list
# Ensure equal distribution across locations and validity conditions
location_names = ['near_left', 'near_right', 'far_left', 'far_right']
trials_per_location = NUMBER_OF_TRIALS // 4
trials_per_validity = NUMBER_OF_TRIALS // 4

# Create sound locations (equal distribution)
sound_locations = []
for loc in location_names:
    sound_locations.extend([loc] * trials_per_location)

# Create validity conditions (equal distribution across 4 conditions)
# Valid, Azimuth Invalid, Distance Invalid, Double Invalid
dot_locations_list = []
for sound_loc in sound_locations:
    sound_distance, sound_azimuth = sound_loc.split('_')
    # Generate one trial of each validity condition for this sound location
    if sound_loc == 'near_left':
        valid_dot = 'near_left'
        azimuth_invalid_dot = 'near_right'
        distance_invalid_dot = 'far_left'
        double_invalid_dot = 'far_right'
    elif sound_loc == 'near_right':
        valid_dot = 'near_right'
        azimuth_invalid_dot = 'near_left'
        distance_invalid_dot = 'far_right'
        double_invalid_dot = 'far_left'
    elif sound_loc == 'far_left':
        valid_dot = 'far_left'
        azimuth_invalid_dot = 'far_right'
        distance_invalid_dot = 'near_left'
        double_invalid_dot = 'near_right'
    else:  # far_right
        valid_dot = 'far_right'
        azimuth_invalid_dot = 'far_left'
        distance_invalid_dot = 'near_right'
        double_invalid_dot = 'near_left'
    
    dot_locations_list.extend([valid_dot, azimuth_invalid_dot, distance_invalid_dot, double_invalid_dot])

# Shuffle sound locations and dot locations independently
random.shuffle(sound_locations)
random.shuffle(dot_locations_list)

# Create results dataframe
results = pd.DataFrame(columns=[
    'trial_number',
    'sound_location',
    'dot_location',
    'validity_condition',
    'sound_file',
    'response',
    'correct',
    'response_time'
])


# Run trials
for trial_num in range(NUMBER_OF_TRIALS):
    # Get current trial parameters
    sound_location = sound_locations[trial_num]
    dot_location = dot_locations_list[trial_num]
    validity_condition = determine_validity_condition(sound_location, dot_location)
    
    # Display fixation cross for 500ms
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)
    
    # Load and play sound cue (only SOUND_CUE_DURATION)
    sound_file = f'{AUDIO_FILE_PREFIX}_{AUDIO_FILE_MAPPING[sound_location]}.wav'
    play_cue(sound_location)

    # Cue-to-target ISI
    win.flip()
    core.wait(CUE_TO_DOT_ISI)

    # Display dot at appropriate location for DOT_DURATION
    dot_pos = LOCATIONS[dot_location]
    dot.pos = dot_pos
    dot.draw()
    win.flip()
    core.wait(DOT_DURATION)

    # Clear screen and wait for response
    win.flip()

    # Record response start time
    response_start = core.getTime()
    event.clearEvents()
    kb.clearEvents()
    response = None
    response_time = None

    # Wait for arrow key response (do not advance trial until a response is made)
    while response is None:
        response = get_direction_response()
        if response is not None:
            response_time = core.getTime() - response_start

    # Check if response was correct
    if response == 'escape':
        break

    correct = response == dot_location if response else False

    # No on-screen feedback; keep timing equivalent to previous feedback period
    core.wait(0.5)

    # Clear screen
    win.flip()

    # Inter-trial interval
    core.wait(INTER_TRIAL_INTERVAL)

    # Store trial data
    results.loc[trial_num] = {
        'trial_number': trial_num + 1,
        'sound_location': sound_location,
        'dot_location': dot_location,
        'validity_condition': validity_condition,
        'sound_file': sound_file,
        'response': response,
        'correct': correct,
        'response_time': response_time
    }

# Save results
results_file = os.path.join(results_dir, f'{participant_id}_results.csv')
results.to_csv(results_file, index=False)
print(f"Results saved to {results_file}")

# End screen
end_text = visual.TextStim(
    win,
    text="Experiment complete!\n\nThank you for participating.",
    color='white',
    height=30
)
end_text.draw()
win.flip()
core.wait(2)

# Cleanup
sd.stop()
cue_stream.stop()
cue_stream.close()
win.close()
core.quit()

