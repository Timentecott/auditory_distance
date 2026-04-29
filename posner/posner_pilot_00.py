from psychopy import visual, event, core
from psychopy.hardware import keyboard
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
import random
import os

# Experiment parameters
NUMBER_OF_TRIALS = 64  # Must be divisible by 8 (4 locations x 2 validity types)
FIXATION_DURATION = 0.5  # 500ms
SOUND_CUE_DURATION = 0.1  # 100ms
CUE_TO_DOT_ISI = 0.1  # 100ms
DOT_DURATION = 0.1  # 100ms
INTER_TRIAL_INTERVAL = 1.0  # seconds
RESPONSE_TIMEOUT = 3.0  # Maximum time to wait for response in seconds
# Set to an integer output device index (from check_input_output_index.py output list).
AUDIO_OUTPUT_DEVICE_INDEX = 4

# Screen coordinates for locations
DISTANCE_FROM_CENTER = 200  # pixels
LOCATIONS = {
    'left': (-DISTANCE_FROM_CENTER, 0),
    'right': (DISTANCE_FROM_CENTER, 0),
    'up': (0, DISTANCE_FROM_CENTER),
    'down': (0, -DISTANCE_FROM_CENTER)
}

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(base_dir, 'audio_stimuli', 'Localised')
soundfile = 'pink_noise_48k_30s_300_8000hz'

START_KEYS = {'num_5', 'num5', 'kp_5', 'numpad5', '5', 'clear'}

# Accept both arrow keys and common numpad key names across keyboards/backends.
RESPONSE_KEY_MAP = {
    'left': 'left',
    'right': 'right',
    'up': 'up',
    'down': 'down',
    'num_4': 'left',
    'num4': 'left',
    'kp_4': 'left',
    'numpad4': 'left',
    '4': 'left',
    'num_6': 'right',
    'num6': 'right',
    'kp_6': 'right',
    'numpad6': 'right',
    '6': 'right',
    'num_8': 'up',
    'num8': 'up',
    'kp_8': 'up',
    'numpad8': 'up',
    '8': 'up',
    'num_2': 'down',
    'num2': 'down',
    'kp_2': 'down',
    'numpad2': 'down',
    '2': 'down',
}


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
    """Play a cue briefly using sounddevice on the configured output device."""
    cue_audio, cue_sr = cue_sounds[location_name]
    sd.stop()
    sd.play(cue_audio, samplerate=cue_sr, device=AUDIO_OUTPUT_DEVICE_INDEX)
    core.wait(SOUND_CUE_DURATION)
    sd.stop()


# Create PsychoPy window
win = visual.Window(
    size=[1920, 1080],
    fullscr=True,
    color='gray',
    units='pix'
)
win.mouseVisible = False
kb = keyboard.Keyboard()

# Preload cue sounds once to reduce onset latency during trials.
cue_sounds = {}
for loc in LOCATIONS:
    cue_path = os.path.join(audio_dir, f'{soundfile}_{loc}.wav')
    if not os.path.exists(cue_path):
        raise FileNotFoundError(f"Missing audio cue file: {cue_path}")
    cue_audio, cue_sr = sf.read(cue_path, dtype='float32')
    cue_sounds[loc] = (cue_audio, cue_sr)

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
    text="You will hear a sound, then see a dot.\nPress an arrow key on the numpad (LEFT, RIGHT, UP, DOWN) to indicate where the dot is as quickly as possible.Only use one finger for the duration of the experiment\n\nPress 5 on the numpad to have a practice.",
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
    text="practice complete, please press 5 to start the main experiment, you will not get feedback in the main experiment",
    color='white',
    height=30,
    wrapWidth=1000
)



# Display practice instructions
practice_instructions.draw()
practice_image.draw()
win.flip()
wait_for_start_key()

# do 4 practice trials, including feedback for each one
practice_location_names = ['left', 'right', 'up', 'down']
practice_trials = [
    {'sound_location': loc, 'is_valid': (i % 2 == 0)}
    for i, loc in enumerate(practice_location_names)
]
random.shuffle(practice_trials)

for practice_trial in practice_trials:
    sound_location = practice_trial['sound_location']
    is_valid = practice_trial['is_valid']

    if is_valid:
        dot_location = sound_location
    else:
        if sound_location == 'left':
            dot_location = 'right'
        elif sound_location == 'right':
            dot_location = 'left'
        elif sound_location == 'up':
            dot_location = 'down'
        elif sound_location == 'down':
            dot_location = 'up'

    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)

    sound_file = f'{soundfile}_{sound_location}.wav'
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
# Ensure equal distribution across locations and validity
location_names = ['left', 'right', 'up', 'down']
trials_per_location = NUMBER_OF_TRIALS // 4
trials_per_validity = NUMBER_OF_TRIALS // 2

# Create sound locations (equal distribution)
sound_locations = []
for loc in location_names:
    sound_locations.extend([loc] * trials_per_location)

# Create validity list (equal distribution)
trial_validity = [True] * trials_per_validity + [False] * trials_per_validity

# Shuffle independently
random.shuffle(sound_locations)
random.shuffle(trial_validity)

# Create results dataframe
results = pd.DataFrame(columns=[
    'trial_number',
    'sound_location',
    'dot_location',
    'is_valid',
    'sound_file',
    'response',
    'correct',
    'response_time'
])

# Run trials
for trial_num in range(NUMBER_OF_TRIALS):
    # Get current trial parameters
    sound_location = sound_locations[trial_num]
    is_valid = trial_validity[trial_num]
    
    # Determine dot location based on validity
    if is_valid:
        dot_location = sound_location
    else:
        # Invalid trial: opposite location
        if sound_location == 'left':
            dot_location = 'right'
        elif sound_location == 'right':
            dot_location = 'left'
        elif sound_location == 'up':
            dot_location = 'down'
        elif sound_location == 'down':
            dot_location = 'up'
    
    # Display fixation cross for 500ms
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)
    
    # Load and play sound cue (only SOUND_CUE_DURATION)
    sound_file = f'{soundfile}_{sound_location}.wav'
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
        'is_valid': is_valid,
        'sound_file': sound_file,
        'response': response,
        'correct': correct,
        'response_time': response_time
    }

# Save results
results_dir = os.path.join(base_dir, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
results_file = os.path.join(results_dir, f'posner_pilot_{timestamp}.csv')
results.to_csv(results_file, index=False)

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
win.close()
core.quit()

