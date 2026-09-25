#far is red cable (right speaker)
#set up stimuli - put localised stimuli into 'localised' folder 
#check line 22 - ISI or stimulus variation? 
#check audio output
#check number of trials

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
from scipy import signal
import serial
import serial.tools.list_ports
import time


def ensure_stereo(audio):
    """Force audio to stereo (L/R) for headphone playback."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.column_stack([audio[:, 0], audio[:, 0]])
    return audio[:, :2]


def route_to_asio_channels(audio, presentation_type, sound_location='near'):
    """Route audio to ASIO channels based on presentation type and location.

    Channels 0-1: Headphones (binaural in-situ/ex-situ)
    Channels 2-3: Loudspeaker (near/far)

    Args:
        audio: Audio array (will be converted to appropriate format)
        presentation_type: 'in_situ', 'ex_situ', or 'loudspeaker'
        sound_location: 'near' or 'far' (only used for loudspeaker)

    Returns:
        4-channel ASIO routed audio
    """
    routed = np.zeros((audio.shape[0], 4), dtype=np.float32)

    if presentation_type in ['in_situ', 'ex_situ']:
        # Headphones: stereo on channels 0-1
        stereo_audio = ensure_stereo(audio)
        routed[:, 0:2] = stereo_audio[:, :2]
    elif presentation_type == 'loudspeaker':
        # Loudspeaker: mono on channel 2 (near) or channel 3 (far)
        mono_audio = audio[:, 0] if audio.ndim > 1 else audio
        channel_idx = 3 if sound_location == 'far' else 2
        routed[:, channel_idx] = mono_audio
    else:
        raise ValueError(f"Unknown presentation_type: {presentation_type}")

    return routed


# Experiment Design Toggle
# Set to True to vary ISI across blocks (all loudspeaker)
# Set to False to vary presentation type across blocks (loudspeaker, in-situ, ex-situ)
VARY_ISI_BY_BLOCK = True

# Experiment parameters
NUMBER_OF_TRIALS = 120  # 3 blocks x 12 trials
FIXATION_DURATION = 0.0  # seconds
SOUND_CUE_DURATION = 0.6  # seconds - play full audio for 0.6 seconds
SILENCE_PAD_SECONDS = 0.0  # seconds of silence padding before/after audio (matches stimulus generation)
SKIP_SILENCE_PAD = False  # Skip the leading silence padding when playing cues
CUE_TO_DOT_ISI = 0  # seconds - start the cue-to-dot interval 0.3 seconds into audio playback
LED_FLASH_DURATION = 0.2  # seconds (was DOT_DURATION, now for LED flash)
LED_SERIAL_LATENCY = 0  # seconds - delay to compensate for Pico serial latency light to sound
INTER_TRIAL_INTERVAL = 1.5  # seconds
RESPONSE_TIMEOUT = 3.0  # Maximum time to wait for response in seconds

# CUE_TO_DOT_ISI values for each block (used only if VARY_ISI_BY_BLOCK=True)
CUE_TO_DOT_ISI_BY_BLOCK = [0, 0.1, 0.2]  # seconds for blocks 1, 2, 3. Note this was 0.2, 0.275, 0.35, changing to 0.2 and down

# Presentation types for each block (used only if VARY_ISI_BY_BLOCK=False)
PRESENTATION_TYPES_BY_BLOCK = ['loudspeaker', 'in_situ', 'ex_situ']  # for blocks 1, 2, 3

# Raspberry Pi Pico communication parameters
PICO_BAUD_RATE = 115200
PICO_TIMEOUT = 2.0  # seconds
# Set to the ASIO aggregate output device index that exposes 4 output channels.
# Channels 0-1: Headphones (binaural in-situ/ex-situ)
# Channels 2-3: Loudspeaker (near/far)
AUDIO_OUTPUT_DEVICE_INDEX = 12

# Presentation types (in situ vs ex situ)
PRESENTATION_TYPES = ['loudspeaker', 'in_situ', 'ex_situ']

# Screen coordinates for locations
DISTANCE_FROM_CENTER = 200  # pixels
LOCATIONS = {
    'far': (0, DISTANCE_FROM_CENTER),
    'near': (0, -DISTANCE_FROM_CENTER),
}

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
audio_stimuli_dir = os.path.join(base_dir, 'audio_stimuli')
audio_dir = os.path.join(audio_stimuli_dir, 'Localised')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Audio file naming prefix
AUDIO_FILE_PREFIX = 'short_tap'

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
    ('in_situ', 'near'): 'in-situ-near',
    ('in_situ', 'far'): 'in-situ-far',
    ('ex_situ', 'near'): 'ex-situ-near',
    ('ex_situ', 'far'): 'ex-situ-far',
    ('loudspeaker', 'near'): 'loudspeaker',  # Uses short_tap.wav via special handling
    ('loudspeaker', 'far'): 'loudspeaker',   # Uses short_tap.wav via special handling
}

# Verify files exist (skip loudspeaker as it uses special file)
missing = []
for (presentation, location), suffix in AUDIO_FILE_MAPPING.items():
    if presentation == 'loudspeaker':
        continue  # Loudspeaker handled separately
    p = os.path.join(audio_dir, f"{AUDIO_FILE_PREFIX}_{suffix}.wav")
    if not os.path.exists(p):
        missing.append(p)

# Verify short_tap_loudspeaker.wav exists for loudspeaker
loudspeaker_path = os.path.join(audio_dir, f"{AUDIO_FILE_PREFIX}_loudspeaker.wav")
if not os.path.exists(loudspeaker_path):
    missing.append(loudspeaker_path)

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
    'audio_duration_samples': 0,
    'presentation_type': 'in_situ',
    'sound_location': 'near'
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

        # Accept any key to continue
        if key_names:
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


# Raspberry Pi Pico communication functions
pico_serial = None

def find_pico_port():
    """Auto-detect Raspberry Pi Pico COM port. Returns port name or None if not found."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Common Pico identifiers
        if 'Pico' in port.description or 'RP2040' in port.description or 'USB' in port.description:
            return port.device
    # If no specific Pico found, try common ports
    if ports:
        return ports[0].device
    return None


def connect_pico():
    """Establish serial connection to Raspberry Pi Pico."""
    global pico_serial
    try:
        port = find_pico_port()
        if port is None:
            print("[PICO] No Raspberry Pi Pico port found!")
            return False

        pico_serial = serial.Serial(port, PICO_BAUD_RATE, timeout=PICO_TIMEOUT)
        time.sleep(2)  # Wait for Pico to initialize
        print(f"[PICO] Connected to Raspberry Pi Pico on port {port}")
        return True
    except Exception as e:
        print(f"[PICO] Failed to connect: {e}")
        return False


def disconnect_pico():
    """Close serial connection to Raspberry Pi Pico."""
    global pico_serial
    if pico_serial and pico_serial.is_open:
        pico_serial.close()
        print("[PICO] Disconnected from Raspberry Pi Pico")


def send_led_command(location):
    """Send LED trigger command to Pico.

    Args:
        location: 'near' or 'far'

    Returns:
        True if command sent successfully, False otherwise
    """
    if not pico_serial or not pico_serial.is_open:
        print("[PICO] Serial port not open!")
        return False

    try:
        # Map location to LED command
        # Pin 15 = near, Pin 16 = far
        # '1' = Pin 15 ON, '0' = Pin 15 OFF
        # '3' = Pin 16 ON, '2' = Pin 16 OFF
        if location == 'near':
            command = '1'  # Pin 15 ON
        elif location == 'far':
            command = '3'  # Pin 16 ON
        else:
            print(f"[PICO] Unknown location: {location}")
            return False

        pico_serial.write(command.encode())
        print(f"[PICO] Sent command for {location}: {command}")
        return True
    except Exception as e:
        print(f"[PICO] Failed to send command: {e}")
        return False


def wait_for_led_complete():
    """Wait for LED flash to complete.

    The Pico flashes the LED for LED_FLASH_DURATION, so we just wait
    for that duration plus a small buffer.

    Returns:
        True (always succeeds since it's just a timer)
    """
    if not pico_serial or not pico_serial.is_open:
        print("[PICO] Serial port not open!")
        return False

    try:
        # Wait for the LED flash duration
        time.sleep(LED_FLASH_DURATION + 0.1)  # Add small buffer
        print("[PICO] LED flash complete")
        return True
    except Exception as e:
        print(f"[PICO] Error during LED wait: {e}")
        return False


def send_led_off_command(location):
    """Send LED OFF command to turn off the LED after flash.

    Args:
        location: 'near' or 'far'

    Returns:
        True if command sent successfully, False otherwise
    """
    if not pico_serial or not pico_serial.is_open:
        print("[PICO] Serial port not open!")
        return False

    try:
        # Map location to LED OFF command
        # '0' = Pin 15 OFF, '2' = Pin 16 OFF
        if location == 'near':
            command = '0'  # Pin 15 OFF
        elif location == 'far':
            command = '2'  # Pin 16 OFF
        else:
            print(f"[PICO] Unknown location: {location}")
            return False

        pico_serial.write(command.encode())
        print(f"[PICO] Sent OFF command for {location}: {command}")
        return True
    except Exception as e:
        print(f"[PICO] Failed to send OFF command: {e}")
        return False


def trigger_led_flash(location):
    """Trigger LED flash and wait for completion.

    Args:
        location: 'near' or 'far'

    Returns:
        True if LED flash completed successfully
    """
    if not send_led_command(location):
        return False

    # Wait for the flash duration
    time.sleep(LED_FLASH_DURATION)

    # Turn off the LED
    send_led_off_command(location)

    return True


def play_cue(presentation_type, location_name, cue_duration=None):
    """Play a cue using the persistent output stream.

    Args:
        presentation_type: 'in_situ', 'ex_situ', or 'loudspeaker'
        location_name: 'near' or 'far'
    """
    print(f"[AUDIO] Playing cue: {presentation_type} {location_name}, duration: {SOUND_CUE_DURATION}s")

    cue_audio, cue_sr = cue_sounds[(presentation_type, location_name)]

    # Calculate start sample, skipping silence padding if enabled
    if SKIP_SILENCE_PAD:
        start_sample = int(round(cue_sr * SILENCE_PAD_SECONDS))
    else:
        start_sample = 0

    cue_samples = int(round(cue_sr * SOUND_CUE_DURATION))
    end_sample = start_sample + cue_samples

    # Extract audio segment, handling the case where we might go past the end
    cue_audio = cue_audio[start_sample:min(end_sample, cue_audio.shape[0])]
    print(f"[AUDIO] Cue samples: {cue_samples}, audio shape after cut: {cue_audio.shape}")

    # If sample rate doesn't match stream sample rate, resample
    if cue_sr != 44100:
        ratio = 44100 / cue_sr
        n_samples = int(round(cue_audio.shape[0] * ratio))
        if cue_audio.ndim == 1:
            cue_audio = signal.resample(cue_audio, n_samples)
        else:
            resampled = []
            for ch in range(cue_audio.shape[1]):
                resampled.append(signal.resample(cue_audio[:, ch], n_samples))
            cue_audio = np.column_stack(resampled)

    with cue_state_lock:
        cue_playback_state['audio'] = np.asarray(cue_audio, dtype=np.float32)
        cue_playback_state['pos'] = 0
        cue_playback_state['audio_duration_samples'] = cue_audio.shape[0]
        cue_playback_state['presentation_type'] = presentation_type
        cue_playback_state['sound_location'] = location_name
        print(f"[AUDIO] Set playback state: duration={cue_audio.shape[0]}, audio_min={np.min(cue_audio)}, audio_max={np.max(cue_audio)}")

    while True:
        with cue_state_lock:
            if cue_playback_state['pos'] >= cue_playback_state['audio_duration_samples']:
                print(f"[AUDIO] Playback complete")
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

# Initialize Pico connection
if not connect_pico():
    print("[WARNING] Pico connection failed. Experiment may not function properly.")
    # Continue anyway - tests can run without Pico for development


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
short_tap_audio = None
short_tap_sr = None

for presentation in PRESENTATION_TYPES:
    for loc in LOCATIONS:
        if presentation == 'loudspeaker':
            # Load short_tap_loudspeaker.wav once for all loudspeaker trials
            if short_tap_audio is None:
                short_tap_path = os.path.join(audio_dir, f'{AUDIO_FILE_PREFIX}_loudspeaker.wav')
                short_tap_audio, short_tap_sr = sf.read(short_tap_path, dtype='float32')
                print(f"Loaded {short_tap_path}: shape={short_tap_audio.shape}, sr={short_tap_sr}Hz")
            cue_sounds[(presentation, loc)] = (short_tap_audio, short_tap_sr)
        else:
            audio_suffix = AUDIO_FILE_MAPPING[(presentation, loc)]
            cue_path = os.path.join(audio_dir, f'{AUDIO_FILE_PREFIX}_{audio_suffix}.wav')
            if not os.path.exists(cue_path):
                raise FileNotFoundError(f"Missing audio cue file: {cue_path}")
            cue_audio, cue_sr = sf.read(cue_path, dtype='float32')
            print(f"[DEBUG] Raw file load: {cue_path}, shape={cue_audio.shape}, min={np.min(cue_audio):.8f}, max={np.max(cue_audio):.8f}")
            # Ensure headphone audio is stereo
            cue_audio = ensure_stereo(cue_audio)
            print(f"Loaded {cue_path}: shape={cue_audio.shape}, sr={cue_sr}Hz, min={np.min(cue_audio):.8f}, max={np.max(cue_audio):.8f}")
            cue_sounds[(presentation, loc)] = (cue_audio, cue_sr)

# Verify all audio files loaded
print(f"[DEBUG] Loaded {len(cue_sounds)} audio files:")
for key in sorted(cue_sounds.keys()):
    audio, sr = cue_sounds[key]
    print(f"[DEBUG]   {key}: shape={audio.shape}, sr={sr}Hz, min={np.min(audio):.8f}, max={np.max(audio):.8f}")


def cue_playback_callback(outdata, frame_count, time_info, status):
    if status:
        print(f"Audio callback status: {status}")

    with cue_state_lock:
        audio = cue_playback_state['audio']
        pos = cue_playback_state['pos']
        audio_duration = cue_playback_state['audio_duration_samples']
        presentation_type = cue_playback_state.get('presentation_type', 'in_situ')
        sound_location = cue_playback_state.get('sound_location', 'near')

        # DEBUG: Print routing info on first frame of each playback
        if pos == 0 and audio is not None:
            print(f"[CALLBACK] Starting playback: presentation_type='{presentation_type}', sound_location='{sound_location}'")
            print(f"[CALLBACK] Audio shape={audio.shape}, dtype={audio.dtype}, min={np.min(audio):.8f}, max={np.max(audio):.8f}")

        if audio is None or pos >= audio_duration:
            outdata[:] = 0
            cue_playback_state['pos'] = pos + frame_count
            return

        end_pos = pos + frame_count
        audio_frame = audio[pos:min(end_pos, audio_duration)]

        # If we got an empty frame, fill with silence
        if audio_frame.size == 0:
            outdata[:] = 0
            cue_playback_state['pos'] = end_pos
            return

        if pos == 0:
            print(f"[CALLBACK] First audio_frame: shape={audio_frame.shape}, dtype={audio_frame.dtype}, min={np.min(audio_frame):.8f}, max={np.max(audio_frame):.8f}")

        # Ensure audio_frame is in the correct format (stereo for headphones, can be mono/stereo for loudspeaker)
        if audio_frame.ndim == 1:
            audio_frame = np.column_stack([audio_frame, audio_frame])
        elif audio_frame.ndim > 2:
            audio_frame = audio_frame[:, :2]

        if pos == 0:
            print(f"[CALLBACK] After stereo conversion: shape={audio_frame.shape}, dtype={audio_frame.dtype}, min={np.min(audio_frame):.8f}, max={np.max(audio_frame):.8f}")

        # Pad audio frame to match requested frame_count if needed
        if audio_frame.shape[0] < frame_count:
            n_channels = audio_frame.shape[1] if audio_frame.ndim > 1 else 1
            if audio_frame.ndim == 1:
                audio_frame = audio_frame[:, np.newaxis]
            padding = np.zeros((frame_count - audio_frame.shape[0], n_channels), dtype=np.float32)
            audio_frame = np.vstack([audio_frame, padding])

        # Route to ASIO channels
        routed = route_to_asio_channels(audio_frame, presentation_type, sound_location)

        if pos == 0:
            max_ch = [np.max(np.abs(routed[:, i])) for i in range(4)]
            print(f"[ROUTING] {presentation_type} {sound_location}: ch0={max_ch[0]:.4f}, ch1={max_ch[1]:.4f}, ch2={max_ch[2]:.4f}, ch3={max_ch[3]:.4f}")

        outdata[:] = routed
        cue_playback_state['pos'] = min(end_pos, audio_duration)

cue_stream = sd.OutputStream(
    samplerate=44100,
    device=AUDIO_OUTPUT_DEVICE_INDEX,
    channels=4,
    dtype='float32',
    callback=cue_playback_callback,
    latency='low',
)
cue_stream.start()
print(f"[STREAM] Audio stream started on device {AUDIO_OUTPUT_DEVICE_INDEX}, stream active: {cue_stream.active}")

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
    """Draw a dot that is equidistant from the fixation cross.
    Both 'near' and 'far' dots are the same size and distance from fixation.
    They are positioned above (far) and below (near) the fixation point."""
    # Use far depth for size (consistent for both dots)
    fixed_depth_for_size = 0.82
    _, _, r_fixed, _, _ = project_point_to_floor(elements, fixed_depth_for_size)
    r = r_fixed

    # Get fixation cross position (center of screen)
    fix_x, fix_y = fixation.pos

    # Calculate a reasonable distance offset from the fixation cross
    # Use the radius as a baseline - the dot should be about 10-11 radii away
    distance_from_fixation = r * 10.5

    # Position dots equidistant from fixation on vertical axis
    x = fix_x  # Both dots stay horizontally centered
    if distance_label == 'far':
        y = fix_y + distance_from_fixation  # Below fixation (far)
    else:  # 'near'
        y = fix_y - distance_from_fixation  # Above fixation (near)

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
    text="Look towards you left, between the two speakers, at the fixation point. \nYou will hear a sound, then see a light flash.\nPress up if the light is on the far speaker or down if it's the near speaker.\nOnly use one finger for the duration of the experiment\n\nPlease press any key to have a practice.",
    color='white',
    height=30,
    wrapWidth=1000,
    pos=(0, 180)
)
instructions = visual.TextStim(
    win,
    text="practice complete, remember to press the button according to where you see the light and ignore the sound. \nRespond as fast as you can. \nTim will now leave the room, then please press any key to start the main experiment, you will not get feedback in the main experiment",
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
calibration_audio_path = r'C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\pink_noise_48k_30s_300_8000hz.wav'
if not os.path.exists(calibration_audio_path):
    raise FileNotFoundError(f"Calibration audio file not found: {calibration_audio_path}")

calibration_audio, calibration_sr = sf.read(calibration_audio_path, dtype='float32')
print(f"[DEBUG] calibration_audio shape: {calibration_audio.shape}, sr={calibration_sr}")

# Resample to match stream sample rate (44100 Hz)
if calibration_sr != 44100:
    ratio = 44100 / calibration_sr
    n_samples = int(round(calibration_audio.shape[0] * ratio))
    if calibration_audio.ndim == 1:
        calibration_audio = signal.resample(calibration_audio, n_samples)
    else:
        # Resample each channel
        resampled = []
        for ch in range(calibration_audio.shape[1]):
            resampled.append(signal.resample(calibration_audio[:, ch], n_samples))
        calibration_audio = np.column_stack(resampled)
    print(f"[DEBUG] Resampled to 44100 Hz: new shape={calibration_audio.shape}")

# Convert stereo to mono by taking average
if calibration_audio.ndim > 1 and calibration_audio.shape[1] > 1:
    calibration_audio = np.mean(calibration_audio, axis=1)
    print(f"[DEBUG] Converted to mono: shape={calibration_audio.shape}")

calibration_duration_seconds = len(calibration_audio) / 44100

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
    cue_playback_state['presentation_type'] = 'loudspeaker'
    cue_playback_state['sound_location'] = 'near'

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
            cue_playback_state['presentation_type'] = 'loudspeaker'
            cue_playback_state['sound_location'] = 'near'

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
win.flip()
wait_for_start_key()

# do 4 practice trials, one from each condition: [in_situ, valid], [in_situ, invalid], [ex_situ, valid], [ex_situ, invalid]
practice_trials = [
    {'presentation_type': 'loudspeaker', 'sound_location': 'near', 'dot_location': 'near'},    # in_situ, valid
    {'presentation_type': 'loudspeaker', 'sound_location': 'far', 'dot_location': 'far'},     # in_situ, invalid
    {'presentation_type': 'loudspeaker', 'sound_location': 'near', 'dot_location': 'far'},    # ex_situ, valid
    {'presentation_type': 'loudspeaker', 'sound_location': 'far', 'dot_location': 'near'},     # ex_situ, invalid
    {'presentation_type': 'loudspeaker', 'sound_location': 'near', 'dot_location': 'far'},    # ex_situ, valid
    {'presentation_type': 'loudspeaker', 'sound_location': 'far', 'dot_location': 'near'},     # ex_situ, invalid
]

random.shuffle(practice_trials)

for practice_trial in practice_trials:
    presentation_type = practice_trial['presentation_type']
    sound_location = practice_trial['sound_location']
    dot_location = practice_trial['dot_location']
    validity_condition = determine_validity_condition(sound_location, dot_location)

    # Display blank screen for 500ms during fixation period
    win.flip()
    core.wait(FIXATION_DURATION)

    # Record the time when audio playback starts
    play_cue_start = core.getTime()
    play_cue(presentation_type, sound_location)

    # Wait for cue-to-dot ISI (sound continues playing in background)
    # The cue-to-dot interval is measured from when audio started playing
    win.flip()
    while core.getTime() - play_cue_start < CUE_TO_DOT_ISI:
        core.wait(0.01)

    # Trigger LED flash at target location for practice
    trigger_led_flash(dot_location)

    # Record response start time before displaying dot
    event.clearEvents()
    kb.clearEvents()
    response = None

    # Keep showing blank screen during LED flash duration (sound continues)
    led_flash_end_time = core.getTime() + LED_FLASH_DURATION
    while core.getTime() < led_flash_end_time:
        win.flip()

        # Check for response during LED flash
        if response is None:
            response = get_direction_response()

        core.wait(0.01)  # Small wait to prevent excessive CPU usage

    # After the LED flash, show blank screen while waiting for response
    win.flip()

    # Wait for response if none was made during LED flash
    while response is None:
        response = get_direction_response()

    if response == 'escape':
        win.close()
        core.quit()

    correct = response == dot_location

    # Display colored rectangle feedback (half screen)
    feedback_rect = visual.Rect(
        win,
        width=win.size[0],
        height=win.size[1] // 2,
        fillColor='green' if correct else 'red',
        lineColor=None,
        pos=(0, 0)
    )
    feedback_text = visual.TextStim(
        win,
        text='Correct' if correct else 'Incorrect',
        color='black',
        height=60
    )
    feedback_rect.draw()
    feedback_text.draw()
    win.flip()
    core.wait(0.5)

    # Blank screen during inter-trial interval
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

def generate_blocked_trial_list(n_trials):
    """
    Generate a blocked trial list with loudspeaker block first, then ex-situ, then in-situ.

    Conditions within each block (2 total):
    - valid (sound and dot at same location)
    - invalid (sound and dot at different locations)

    Each third of trials is one presentation type.
    Within each block, trials are balanced and randomized.

    Args:
        n_trials: Total number of trials (must be divisible by 3)

    Returns:
        List of trial dicts with keys: presentation_type, sound_location, dot_location
    """
    if n_trials % 3 != 0:
        raise ValueError("n_trials must be divisible by 3 for three equal blocks")

    trials_per_block = n_trials // 3
    if trials_per_block % 2 != 0:
        raise ValueError("Each block must be divisible by 2 for valid/invalid balance")

    # Block 1: loudspeaker, Block 2: ex-situ, Block 3: in-situ
    presentation_order = ['loudspeaker', 'ex_situ', 'in_situ']

    trial_list = []

    for presentation_type in presentation_order:
        trials_per_validity = trials_per_block // 2

        # Create conditions: valid and invalid
        conditions = ['valid'] * trials_per_validity + ['invalid'] * trials_per_validity

        # Shuffle within block
        random.shuffle(conditions)

        for validity in conditions:
            # Randomly choose which location pair for sound/dot
            if random.choice([True, False]):
                sound_location = 'near'
                dot_location = 'near' if validity == 'valid' else 'far'
            else:
                sound_location = 'far'
                dot_location = 'far' if validity == 'valid' else 'near'

            trial_list.append({
                'presentation_type': presentation_type,
                'sound_location': sound_location,
                'dot_location': dot_location
            })

    return trial_list


def generate_trial_list(n_trials, vary_isi=True):
    """
    Generate trials with either varying ISI or varying presentation type.

    Args:
        n_trials: Total number of trials (must be divisible by 3 for 3 blocks)
        vary_isi: If True, vary CUE_TO_DOT_ISI across blocks (all loudspeaker).
                  If False, vary presentation_type across blocks (loudspeaker, in_situ, ex_situ).

    Returns:
        List of trial dicts with keys: presentation_type, sound_location, dot_location, cue_to_dot_isi
    """
    if vary_isi:
        block_params = CUE_TO_DOT_ISI_BY_BLOCK
        param_key = 'cue_to_dot_isi'
    else:
        block_params = PRESENTATION_TYPES_BY_BLOCK
        param_key = 'presentation_type'

    num_blocks = len(block_params)
    if n_trials % num_blocks != 0:
        raise ValueError(f"n_trials ({n_trials}) must be divisible by number of blocks ({num_blocks})")

    trials_per_block = n_trials // num_blocks
    if trials_per_block % 2 != 0:
        raise ValueError(f"Each block must have even number of trials for valid/invalid balance, got {trials_per_block}")

    trial_list = []

    for block_idx, block_param in enumerate(block_params):
        trials_per_validity = trials_per_block // 2

        # Create conditions: valid and invalid
        conditions = ['valid'] * trials_per_validity + ['invalid'] * trials_per_validity

        # Shuffle within block
        random.shuffle(conditions)

        for validity in conditions:
            # Randomly choose which location pair for sound/dot
            if random.choice([True, False]):
                sound_location = 'near'
                dot_location = 'near' if validity == 'valid' else 'far'
            else:
                sound_location = 'far'
                dot_location = 'far' if validity == 'valid' else 'near'

            trial_dict = {
                'sound_location': sound_location,
                'dot_location': dot_location,
            }

            # Add block-specific parameter
            if vary_isi:
                trial_dict['presentation_type'] = 'loudspeaker'
                trial_dict['cue_to_dot_isi'] = block_param
            else:
                trial_dict['presentation_type'] = block_param
                trial_dict['cue_to_dot_isi'] = CUE_TO_DOT_ISI

            trial_list.append(trial_dict)

    return trial_list



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

# Seed RNG with participant ID for reproducible trial sequences
random.seed(int(participant_id))

trials = generate_trial_list(NUMBER_OF_TRIALS, vary_isi=VARY_ISI_BY_BLOCK)

# DEBUG: Print first few trials to verify presentation_type
print("\n[DEBUG] Trial generation verification:")
print(f"[DEBUG] VARY_ISI_BY_BLOCK = {VARY_ISI_BY_BLOCK}")
for i, trial in enumerate(trials[:3]):
    print(f"[DEBUG] Trial {i}: presentation_type='{trial['presentation_type']}', sound_location='{trial['sound_location']}'")
print()

# Create results dataframe
results = pd.DataFrame(columns=[
    'trial_number',
    'presentation_type',
    'sound_location',
    'dot_location',
    'cue_to_dot_isi',
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
    cue_to_dot_isi = trial.get('cue_to_dot_isi', CUE_TO_DOT_ISI)
    validity_condition = determine_validity_condition(sound_location, dot_location)

    # Display blank screen for 500ms during fixation period
    win.flip()
    core.wait(FIXATION_DURATION)

    # Load and play sound cue (will continue playing in background)
    audio_suffix = AUDIO_FILE_MAPPING[(presentation_type, sound_location)]
    sound_file = f'{AUDIO_FILE_PREFIX}_{audio_suffix}.wav'

    # Record the time when audio playback starts
    play_cue_start = core.getTime()
    play_cue(presentation_type, sound_location)

    # Wait for cue-to-dot ISI (sound continues playing in background)
    # The cue-to-dot interval is measured from when audio started playing
    win.flip()
    while core.getTime() - play_cue_start < cue_to_dot_isi:
        core.wait(0.01)

    # Trigger LED flash at target location
    trigger_led_flash(dot_location)

    # Record response start time
    response_start = core.getTime()
    event.clearEvents()
    kb.clearEvents()
    response = None
    response_time = None

    # Keep showing blank screen during LED flash duration (sound continues)
    led_flash_end_time = response_start + LED_FLASH_DURATION
    while core.getTime() < led_flash_end_time:
        win.flip()

        # Check for response during LED flash
        if response is None:
            response = get_direction_response()
            if response is not None:
                response_time = core.getTime() - response_start

        core.wait(0.01)  # Small wait to prevent excessive CPU usage

    # After LED flash, show blank screen, wait for response if none was given
    win.flip()

    # Wait for arrow key response if none was made during LED flash
    while response is None:
        response = get_direction_response()
        if response is not None:
            response_time = core.getTime() - response_start

    # Check if response was correct
    if response == 'escape':
        break

    correct = response == dot_location if response else False

    # No on-screen feedback; blank screen during inter-trial interval
    core.wait(0.5)

    # Show blank screen during inter-trial interval
    win.flip()
    core.wait(INTER_TRIAL_INTERVAL)

    # Store trial data
    results.loc[trial_num] = {
        'trial_number': trial_num + 1,
        'presentation_type': presentation_type,
        'sound_location': sound_location,
        'dot_location': dot_location,
        'cue_to_dot_isi': cue_to_dot_isi,
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

# Calculate mean response times for each condition
condition_means = {}
condition_stds = {}

if VARY_ISI_BY_BLOCK:
    # Analyze by ISI (with constant presentation type: loudspeaker)
    cue_to_dot_isis = sorted(results['cue_to_dot_isi'].dropna().unique())

    for validity in ['Valid', 'Invalid']:
        for isi_dur in cue_to_dot_isis:
            condition_mask = (results['presentation_type'] == 'loudspeaker') & \
                            (results['validity_condition'] == validity) & \
                            (results['cue_to_dot_isi'] == isi_dur)
            valid_rts = results.loc[condition_mask, 'response_time'].dropna()
            if len(valid_rts) > 0:
                mean_rt = valid_rts.mean()
                std_rt = valid_rts.std()
                condition_means[('loudspeaker', validity, isi_dur)] = mean_rt
                condition_stds[('loudspeaker', validity, isi_dur)] = std_rt
                print(f"loudspeaker {validity} (ISI={isi_dur}s): mean RT = {mean_rt:.4f}s, SD = {std_rt:.4f}s (n={len(valid_rts)})")
            else:
                condition_means[('loudspeaker', validity, isi_dur)] = np.nan
                condition_stds[('loudspeaker', validity, isi_dur)] = np.nan
else:
    # Analyze by presentation type (with constant ISI)
    presentation_types = sorted(results['presentation_type'].dropna().unique())

    for presentation_type in presentation_types:
        for validity in ['Valid', 'Invalid']:
            condition_mask = (results['presentation_type'] == presentation_type) & \
                            (results['validity_condition'] == validity)
            valid_rts = results.loc[condition_mask, 'response_time'].dropna()
            if len(valid_rts) > 0:
                mean_rt = valid_rts.mean()
                std_rt = valid_rts.std()
                condition_means[(presentation_type, validity)] = mean_rt
                condition_stds[(presentation_type, validity)] = std_rt
                print(f"{presentation_type} {validity}: mean RT = {mean_rt:.4f}s, SD = {std_rt:.4f}s (n={len(valid_rts)})")
            else:
                condition_means[(presentation_type, validity)] = np.nan
                condition_stds[(presentation_type, validity)] = np.nan

# Add condition means and SDs as new columns
def get_condition_mean(row):
    if VARY_ISI_BY_BLOCK:
        key = (row['presentation_type'], row['validity_condition'], row['cue_to_dot_isi'])
    else:
        key = (row['presentation_type'], row['validity_condition'])
    return condition_means.get(key, np.nan)

def get_condition_std(row):
    if VARY_ISI_BY_BLOCK:
        key = (row['presentation_type'], row['validity_condition'], row['cue_to_dot_isi'])
    else:
        key = (row['presentation_type'], row['validity_condition'])
    return condition_stds.get(key, np.nan)

results['condition_mean_response_time'] = results.apply(get_condition_mean, axis=1)
results['condition_std_response_time'] = results.apply(get_condition_std, axis=1)

# Save results
results_file = os.path.join(results_dir, f'{participant_id}_results.csv')
results.to_csv(results_file, index=False)
print(f"Results saved to {results_file}")

# Also save a summary of condition means and standard deviations
summary_data = []

if VARY_ISI_BY_BLOCK:
    # Summary by ISI
    cue_to_dot_isis = sorted(results['cue_to_dot_isi'].dropna().unique())
    for validity in ['Valid', 'Invalid']:
        for isi_dur in cue_to_dot_isis:
            key = ('loudspeaker', validity, isi_dur)
            if key in condition_means:
                summary_data.append({
                    'presentation_type': 'loudspeaker',
                    'validity_condition': validity,
                    'cue_to_dot_isi': isi_dur,
                    'mean_response_time': condition_means[key],
                    'std_response_time': condition_stds[key]
                })
else:
    # Summary by presentation type
    presentation_types = sorted(results['presentation_type'].dropna().unique())
    for presentation_type in presentation_types:
        for validity in ['Valid', 'Invalid']:
            key = (presentation_type, validity)
            if key in condition_means:
                summary_data.append({
                    'presentation_type': presentation_type,
                    'validity_condition': validity,
                    'cue_to_dot_isi': CUE_TO_DOT_ISI,
                    'mean_response_time': condition_means[key],
                    'std_response_time': condition_stds[key]
                })

summary_df = pd.DataFrame(summary_data)
summary_file = os.path.join(results_dir, f'{participant_id}_condition_means.csv')
summary_df.to_csv(summary_file, index=False)
print(f"Condition means saved to {summary_file}")

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
disconnect_pico()
sd.stop()
cue_stream.stop()
cue_stream.close()
win.close()
core.quit()

