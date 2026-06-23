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
NUMBER_OF_TRIALS = 64  # Must be divisible by 8 (2 presentation types x 2 validity types x 2 locations)
FIXATION_DURATION = 0.5  # seconds
SOUND_CUE_DURATION = 0.1  # seconds
CUE_TO_DOT_ISI = 0.2  # seconds includes 100ms cue duration
DOT_DURATION = 0.5  # seconds
INTER_TRIAL_INTERVAL = 1.5  # seconds
RESPONSE_TIMEOUT = 3.0  # Maximum time to wait for response in seconds
# Set to the ASIO aggregate output device index that exposes 4 output channels.
# Channels 3-4 are used for headphone playback.
AUDIO_OUTPUT_DEVICE_INDEX = 16

# Presentation types (in situ vs ex situ)
PRESENTATION_TYPES = ['in_situ', 'ex_situ']

# Screen coordinates for locations
DISTANCE_FROM_CENTER = 200  # pixels
LOCATIONS = {
    'far': (0, DISTANCE_FROM_CENTER),
    'near': (0, -DISTANCE_FROM_CENTER),
}

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(base_dir, 'audio_stimuli\localised')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Audio file naming prefix
AUDIO_FILE_PREFIX = 'pink_noise_48k_30s_300_8000hz'

# Helper: find an existing file suffix for a location, prefer left variants
def _find_existing_suffix(location_key):
    # prefer left versions, then generic, then right
    candidates = [f"{location_key}_left", f"{location_key}", f"{location_key}_right"]
    for c in candidates:
        p = os.path.join(audio_dir, f"{AUDIO_FILE_PREFIX}_{c}.wav")
        if os.path.exists(p):
            return c
    return None

# Map logical (presentation_type, location) pairs to actual audio file suffixes (use left variants only)
# Format: (presentation_type, location) -> suffix
AUDIO_FILE_MAPPING = {
    ('in_situ', 'near'): 'in-situ-near',## add more to the suffix once the proper files are created.
    ('in_situ', 'far'): 'in-situ-far',
    ('ex_situ', 'near'): 'ex-situ-near',
    ('ex_situ', 'far'): 'ex-situ-far',
}

# Verify files exist
missing = []
for (presentation, location), suffix in AUDIO_FILE_MAPPING.items():
    p = os.path.join(audio_dir, f"{AUDIO_FILE_PREFIX}_{suffix}.wav")
    if not os.path.exists(p):
        missing.append(p)

if missing:
    raise FileNotFoundError(
        "Missing audio cue files. Expected the following files in audio_stimuli:\n" + "\n".join(missing)
    )

participant_id = ""
demographics = pd.DataFrame(columns=['participant_id'])

START_KEYS = {'num_5', 'num5', 'kp_5', 'numpad5', '5', 'clear'}

# Accept both numpad keys and arrow keys. num 2 => near, num 8 => far
RESPONSE_KEY_MAP = {
    'down': 'near',
    'up': 'far',
    'num_2': 'near',
    'num2': 'near',
    'kp_2': 'near',
    'numpad2': 'near',
    '2': 'near',
    'num_8': 'far',
    'num8': 'far',
    'kp_8': 'far',
    'numpad8': 'far',
    '8': 'far',
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


def play_cue(presentation_type, location_name):
    """Play the first 100 ms of a cue using the persistent output stream.

    Args:
        presentation_type: 'in_situ' or 'ex_situ'
        location_name: 'near' or 'far'
    """
    cue_audio, cue_sr = cue_sounds[(presentation_type, location_name)]
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
for presentation in PRESENTATION_TYPES:
    for loc in LOCATIONS:
        audio_suffix = AUDIO_FILE_MAPPING[(presentation, loc)]
        cue_path = os.path.join(audio_dir, f'{AUDIO_FILE_PREFIX}_{audio_suffix}.wav')
        if not os.path.exists(cue_path):
            raise FileNotFoundError(f"Missing audio cue file: {cue_path}")
        cue_audio, cue_sr = sf.read(cue_path, dtype='float32')
        cue_sounds[(presentation, loc)] = (cue_audio, cue_sr)


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
    fillColor='white',
    pos=(0, -1080 / 4)  # 3/4 way down the screen, based on 1080 pixel height. Will update dynamically below.
)

dot = visual.Circle(
    win,
    radius=20,
    fillColor='white',
    lineColor='white'
)

# --- Enhanced perspective room utilities (more 3D look) ---

def build_perspective_elements(win, n_rows=12, n_cols=12):
    """Build a perspective floor (checkerboard-like tiles) without walls.
    Returns a dict of visual stimuli and parameters.
    """
    w, h = win.size
    half_w, half_h = w / 2.0, h / 2.0

    # Floor limits (fills the lower half/quarter of the screen to horizon)
    # We want the fixation cross (middle of floor) to be 3/4 way down the screen, i.e. at y = -half_h / 2
    # Let's place the floor top (horizon) slightly above that, and bottom at the very bottom
    floor_top_y = -half_h * 0.1
    floor_bottom_y = -half_h
    floor_left_top = -half_w * 0.35
    floor_right_top = half_w * 0.35

    # Generate tiled floor: rows from near (bottom) to far (top), columns across width
    floor_tiles = []
    for row in range(n_rows):
        t0 = row / float(n_rows)
        t1 = (row + 1) / float(n_rows)
        # Y coords for top and bottom of this row
        y0 = floor_top_y * (1 - t0) + floor_bottom_y * t0
        y1 = floor_top_y * (1 - t1) + floor_bottom_y * t1

        # compute left/right x for top and bottom edges of this row
        left_top = floor_left_top * (1 - t0) + (-half_w) * t0
        right_top = floor_right_top * (1 - t0) + (half_w) * t0
        left_bottom = floor_left_top * (1 - t1) + (-half_w) * t1
        right_bottom = floor_right_top * (1 - t1) + (half_w) * t1

        for col in range(n_cols):
            s0 = col / float(n_cols)
            s1 = (col + 1) / float(n_cols)
            # Interpolate along top edge
            x0_left = left_top * (1 - s0) + right_top * s0
            x0_right = left_top * (1 - s1) + right_top * s1
            # Interpolate along bottom edge
            x1_left = left_bottom * (1 - s0) + right_bottom * s0
            x1_right = left_bottom * (1 - s1) + right_bottom * s1

            verts = [(x0_left, y0), (x0_right, y0), (x1_right, y1), (x1_left, y1)]
            # Checker shading depending on row+col and subtle gradient towards back
            parity = (row + col) % 2
            base_shade = 0.20 + 0.45 * (row / float(max(1, n_rows - 1)))
            shade = base_shade - (0.04 if parity == 0 else -0.02)
            shade = float(max(0.05, min(1.0, shade)))
            tile = visual.ShapeStim(win, vertices=verts, closeShape=True, fillColor=(shade, shade, shade), lineColor=None)
            floor_tiles.append(tile)

    return {
        'floor_tiles': floor_tiles,
        'floor_top_y': floor_top_y,
        'floor_bottom_y': floor_bottom_y,
        'floor_left_top': floor_left_top,
        'floor_right_top': floor_right_top,
        'half_w': half_w,
        'half_h': half_h
    }


def draw_perspective_room(elements):
    """Draw the perspective floor as a background. Call before drawing overlays."""
    for tile in elements.get('floor_tiles', []):
        tile.draw()


# Build perspective elements once (replace previous builder call)
perspective_elements = build_perspective_elements(win)

# Dynamically update the fixation position to be exactly in the center of the rendered floor
floor_mid_y = (perspective_elements['floor_top_y'] + perspective_elements['floor_bottom_y']) / 2.0
fixation.pos = (0, floor_mid_y)


def project_point_to_floor(elements, depth_t):
    """Project a normalized depth value in [0,1] (0=near, 1=far) onto screen coordinates and size.
    Returns (x, y, radius, shadow_width, shadow_height)
    """
    bottom_y = elements['floor_bottom_y']
    top_y = elements['floor_top_y']

    # Nonlinear perspective mapping: nearer objects appear much lower on screen
    power = 1.35
    s = depth_t
    # Interpolate between top and bottom with easing to exaggerate foreground
    y = top_y + (bottom_y - top_y) * (1.0 - (s ** (1.0 / power)))

    # lateral x stays centered
    x = 0

    # radius scales with inverse of depth (near larger)
    screen_diag = min(elements['half_w'], elements['half_h'])
    base = screen_diag * 0.06
    radius = max(4, int(base * (1.0 - 0.65 * s)))

    # shadow scales and becomes more elliptical for near objects
    shadow_w = radius * 2.6
    shadow_h = max(2, int(radius * (0.4 + 0.35 * s)))

    return x, y, radius, shadow_w, shadow_h


def draw_dot_in_room(elements, distance_label, color='white'):
    """Draw a dot that convincingly sits on the room floor. distance_label in ('near','far')."""
    # Map labels to normalized depth values
    depth_map = {
        'near': 0.18,
        'far': 0.82
    }
    t = depth_map.get(distance_label, 0.5)
    x, y, r, sw, sh = project_point_to_floor(elements, t)

    # Shadow ellipse drawn on the floor beneath the dot
    n_ellipse = 32
    verts = []
    for i in range(n_ellipse):
        theta = 2.0 * np.pi * i / n_ellipse
        vx = (sw / 2.0) * np.cos(theta)
        vy = (sh / 2.0) * np.sin(theta)
        verts.append((x + vx, y - (r * 0.35) + vy))
    shadow_shape = visual.ShapeStim(win, vertices=verts, closeShape=True, fillColor=(0.06,0.06,0.06), lineColor=None)
    shadow_shape.draw()

    # Draw a slightly larger darker ring to give depth, then bright core
    ring = visual.Circle(win, radius=r * 1.1, edges=64, fillColor=(0.08,0.08,0.08), lineColor=None, pos=(x, y))
    core_dot = visual.Circle(win, radius=r, edges=64, fillColor=color, lineColor='white', pos=(x, y))
    ring.draw()
    core_dot.draw()

    # subtle specular highlight
    highlight = visual.Circle(win, radius=max(1, int(r * 0.22)), edges=32, fillColor=(1,1,1), lineColor=None, pos=(x - r*0.18, y + r*0.18))
    highlight.draw()

    return (x, y, r)


practice_instructions = visual.TextStim(
    win,
    text="You will hear a sound, then see a dot.\nPress 2 on the numpad for NEAR and 8 on the numpad for FAR.\nOnly use one finger for the duration of the experiment\n\nPress 5 on the numpad to have a practice.",
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
    Determine validity for two-location design (near/far).
    Returns 'Valid' if locations match, otherwise 'Invalid'.
    """
    return 'Valid' if sound_loc == dot_loc else 'Invalid'


# Calibration: Play pink noise audio on loop
calibration_text = visual.TextStim(
    win,
    text="Audio Calibration\n\nPress any key when you are ready to proceed.",
    color='white',
    height=30,
    wrapWidth=1200,
    pos=(0, 180)
)

# Load the calibration audio file
calibration_audio_path = r'C:\Users\tim_e\source\repos\auditory_distance\posner\audio_stimuli\pink_noise_48k_30s_300_8000hz.wav'
if not os.path.exists(calibration_audio_path):
    raise FileNotFoundError(f"Calibration audio file not found: {calibration_audio_path}")

calibration_audio, calibration_sr = sf.read(calibration_audio_path, dtype='float32')
calibration_duration_seconds = len(calibration_audio) / calibration_sr

# Display calibration screen and start looping audio
calibration_active = True
calibration_start_time = core.getTime()

event.clearEvents()
kb.clearEvents()

# Initialize audio playback immediately
with cue_state_lock:
    cue_playback_state['audio'] = np.asarray(calibration_audio, dtype=np.float32)
    cue_playback_state['pos'] = 0
    cue_playback_state['audio_duration_samples'] = calibration_audio.shape[0]

# Start playing the audio and loop until key press
while calibration_active:
    calibration_text.draw()
    win.flip()

    # Calculate elapsed time since calibration started
    elapsed = core.getTime() - calibration_start_time
    loop_position = elapsed % calibration_duration_seconds

    # Restart audio playback if we've completed a loop
    if loop_position < 0.05:  # Small threshold to restart
        with cue_state_lock:
            cue_playback_state['audio'] = np.asarray(calibration_audio, dtype=np.float32)
            cue_playback_state['pos'] = 0
            cue_playback_state['audio_duration_samples'] = calibration_audio.shape[0]

    # Check for any key press to stop calibration
    kb_keys = kb.getKeys(waitRelease=False, clear=True)
    event_keys = event.getKeys()

    if kb_keys or event_keys:
        calibration_active = False
        # Stop audio immediately by clearing playback state
        with cue_state_lock:
            cue_playback_state['audio'] = None
            cue_playback_state['pos'] = 0
            cue_playback_state['audio_duration_samples'] = 0

    core.wait(0.01)

# Display practice instructions
practice_instructions.draw()
practice_image.draw()
win.flip()
wait_for_start_key()

# do 4 practice trials, one from each condition: [in_situ, valid], [in_situ, invalid], [ex_situ, valid], [ex_situ, invalid]
practice_trials = [
    {'presentation_type': 'in_situ', 'sound_location': 'near', 'dot_location': 'near'},    # in_situ, valid
    {'presentation_type': 'in_situ', 'sound_location': 'near', 'dot_location': 'far'},     # in_situ, invalid
    {'presentation_type': 'ex_situ', 'sound_location': 'near', 'dot_location': 'near'},    # ex_situ, valid
    {'presentation_type': 'ex_situ', 'sound_location': 'near', 'dot_location': 'far'},     # ex_situ, invalid
]

random.shuffle(practice_trials)

for practice_trial in practice_trials:
    presentation_type = practice_trial['presentation_type']
    sound_location = practice_trial['sound_location']
    dot_location = practice_trial['dot_location']
    validity_condition = determine_validity_condition(sound_location, dot_location)

    # Draw persistent perspective room as background, then fixation on top
    draw_perspective_room(perspective_elements)
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)

    play_cue(presentation_type, sound_location)

    # Keep the room visible during the cue-to-dot interval
    draw_perspective_room(perspective_elements)
    win.flip()
    core.wait(CUE_TO_DOT_ISI)

    # Record response start time before displaying dot
    event.clearEvents()
    kb.clearEvents()
    response = None

    # Draw perspective room and dot according to near/far distance (dot overlays the room)
    # Responses can be made during this period
    dot_display_start = core.getTime()
    while core.getTime() - dot_display_start < DOT_DURATION:
        draw_perspective_room(perspective_elements)
        draw_dot_in_room(perspective_elements, dot_location)
        win.flip()

        # Check for response during dot display
        if response is None:
            response = get_direction_response()

        core.wait(0.01)  # Small wait to prevent excessive CPU usage

    # After the dot disappears, show the room alone while waiting for response
    draw_perspective_room(perspective_elements)
    win.flip()

    # Wait for response if none was made during dot display
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
    # Draw feedback over room
    draw_perspective_room(perspective_elements)
    feedback.draw()
    win.flip()
    core.wait(0.5)

    # Show room during inter-trial interval
    draw_perspective_room(perspective_elements)
    win.flip()
    core.wait(INTER_TRIAL_INTERVAL)

#display instrunctions for main experiment
instructions.draw()
win.flip()
wait_for_start_key()


# Seed the random generator with participant ID for reproducibility
participant_seed = int(participant_id) if participant_id.isdigit() else hash(participant_id) % (2**31)
random.seed(participant_seed)
np.random.seed(participant_seed)

def generate_balanced_trial_list(n_trials):
    """
    Generate a balanced trial list with equal numbers of each condition.

    Conditions (4 total):
    - in_situ + valid
    - in_situ + invalid
    - ex_situ + valid
    - ex_situ + invalid

    No condition repeats more than 3 times consecutively.
    Shuffled randomly using seeded RNG for reproducibility.

    Args:
        n_trials: Total number of trials (must be divisible by 4)

    Returns:
        List of trial dicts with keys: presentation_type, sound_location, dot_location
    """
    if n_trials % 4 != 0:
        raise ValueError("n_trials must be divisible by 4 for equal distribution across 4 conditions")

    trials_per_condition = n_trials // 4

    # Define conditions as tuples (presentation_type, validity)
    conditions = [
        ('in_situ', 'valid'),
        ('in_situ', 'invalid'),
        ('ex_situ', 'valid'),
        ('ex_situ', 'invalid'),
    ]

    # Create base list with equal numbers of each condition
    condition_list = []
    for condition in conditions:
        condition_list.extend([condition] * trials_per_condition)

    # Shuffle with max consecutive constraint
    max_attempts = 10000
    max_consecutive = 3

    for attempt in range(max_attempts):
        random.shuffle(condition_list)

        # Check if shuffle satisfies constraint (no more than 3 consecutive of same condition)
        valid = True
        consecutive_count = 1
        for i in range(1, len(condition_list)):
            if condition_list[i] == condition_list[i-1]:
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    valid = False
                    break
            else:
                consecutive_count = 1

        if valid:
            break

    # Convert conditions to trial dicts
    trial_list = []
    for condition in condition_list:
        presentation_type, validity = condition

        # Randomly choose which location pair for sound/dot
        if random.choice([True, False]):
            sound_location = 'near'
            dot_location = 'near' if validity == 'valid' else 'far'
        else:
            sound_location = 'far'
            dot_location = 'far' if validity == 'valid' else 'near'

        trial = {
            'presentation_type': presentation_type,
            'sound_location': sound_location,
            'dot_location': dot_location,
        }
        trial_list.append(trial)

    return trial_list

trials = generate_balanced_trial_list(NUMBER_OF_TRIALS)

# Create results dataframe
results = pd.DataFrame(columns=[
    'trial_number',
    'presentation_type',
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
    trial = trials[trial_num]
    presentation_type = trial['presentation_type']
    sound_location = trial['sound_location']
    dot_location = trial['dot_location']
    validity_condition = determine_validity_condition(sound_location, dot_location)

    # Display fixation cross for 500ms overlayed on the perspective room
    draw_perspective_room(perspective_elements)
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)

    # Load and play sound cue (only SOUND_CUE_DURATION)
    audio_suffix = AUDIO_FILE_MAPPING[(presentation_type, sound_location)]
    sound_file = f'{AUDIO_FILE_PREFIX}_{audio_suffix}.wav'
    play_cue(presentation_type, sound_location)

    # Cue-to-target ISI - keep room visible
    draw_perspective_room(perspective_elements)
    win.flip()
    core.wait(CUE_TO_DOT_ISI)

    # Record response start time before displaying dot
    response_start = core.getTime()
    event.clearEvents()
    kb.clearEvents()
    response = None
    response_time = None

    # Display dot at appropriate location for DOT_DURATION within the perspective room
    # Responses can be made during this period
    dot_display_start = core.getTime()
    while core.getTime() - dot_display_start < DOT_DURATION:
        draw_perspective_room(perspective_elements)
        draw_dot_in_room(perspective_elements, dot_location)
        win.flip()

        # Check for response during dot display
        if response is None:
            response = get_direction_response()
            if response is not None:
                response_time = core.getTime() - response_start

        core.wait(0.01)  # Small wait to prevent excessive CPU usage

    # After dot disappears, show room alone and wait for response if none was given
    draw_perspective_room(perspective_elements)
    win.flip()

    # Wait for arrow key response if none was made during dot display
    while response is None:
        response = get_direction_response()
        if response is not None:
            response_time = core.getTime() - response_start

    # Check if response was correct
    if response == 'escape':
        break

    correct = response == dot_location if response else False

    # No on-screen feedback; keep timing equivalent to previous feedback period but keep room visible
    core.wait(0.5)

    # Show room during inter-trial interval
    draw_perspective_room(perspective_elements)
    win.flip()
    core.wait(INTER_TRIAL_INTERVAL)

    # Store trial data
    results.loc[trial_num] = {
        'trial_number': trial_num + 1,
        'presentation_type': presentation_type,
        'sound_location': sound_location,
        'dot_location': dot_location,
        'validity_condition': validity_condition,
        'sound_file': sound_file,
        'response': response,
        'correct': correct,
        'response_time': response_time
    }

    # Check for pause breaks at 1/3 and 2/3 completion
    trials_completed = trial_num + 1
    one_third = NUMBER_OF_TRIALS / 3
    two_thirds = (NUMBER_OF_TRIALS * 2) / 3

    if abs(trials_completed - one_third) < 0.5:  # At approximately 1/3
        pause_text = visual.TextStim(
            win,
            text="You are 1/3 of the way through.\n\nPlease take a break and press any key when you are ready to continue.",
            color='white',
            height=30,
            wrapWidth=1200
        )
        pause_text.draw()
        win.flip()
        event.clearEvents()
        kb.clearEvents()
        while True:
            kb_keys = kb.getKeys(waitRelease=False, clear=True)
            event_keys = event.getKeys()
            if kb_keys or event_keys:
                break
            core.wait(0.01)

    elif abs(trials_completed - two_thirds) < 0.5:  # At approximately 2/3
        pause_text = visual.TextStim(
            win,
            text="You are 2/3 of the way through.\n\nPlease take a break and press any key when you are ready to continue.",
            color='white',
            height=30,
            wrapWidth=1200
        )
        pause_text.draw()
        win.flip()
        event.clearEvents()
        kb.clearEvents()
        while True:
            kb_keys = kb.getKeys(waitRelease=False, clear=True)
            event_keys = event.getKeys()
            if kb_keys or event_keys:
                break
            core.wait(0.01)

# Save results
results_file = os.path.join(results_dir, f'{participant_id}_results.csv')
results.to_csv(results_file, index=False)
print(f"Results saved to {results_file}")

# End screen
end_text = visual.TextStim(
    win,
    text="Experiment complete!\n\n your results have been saved.\n\nThank you for participating.",
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

