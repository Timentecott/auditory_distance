from psychopy import visual, event, core
import pandas as pd
import time
import numpy as np
import random
import sounddevice as sd
import soundfile as sf
import os, glob


# Safety limiter for dangerous sound levels
MAX_SPL_DB = 95  # Maximum SPL in dB (85 dB is OSHA safe limit for 8 hours)
# Assuming standard calibration: 0 dBFS = 94 dB SPL
# So max safe level = 94 - (94 - 85) = -9 dBFS
MAX_DBFSsafe = -9.0
MAX_AMP_LINEAR = 10 ** (MAX_DBFSsafe / 20.0)

def apply_safety_limit(audio, max_amplitude=MAX_AMP_LINEAR):
    """
    Apply a hard limiter to prevent dangerously loud playback.
    
    Args:
        audio: Audio array
        max_amplitude: Maximum allowed amplitude (default -9 dBFS ≈ 85 dB SPL)
    
    Returns:
        Limited audio array
    """
    current_max = np.max(np.abs(audio))
    if current_max > max_amplitude:
        scaling_factor = max_amplitude / current_max
        audio = audio * scaling_factor
        print(f"  WARNING: Safety limiter engaged: reduced gain by {20 * np.log10(scaling_factor):.1f} dB")
    return audio


#load headphone stimuli from /localised_stimuli
base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
headphone_dir = os.path.join(base_dir, 'localised_stimuli_b20')
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


#display instructions: 

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
    wrapWidth=1100  # Add this to make lines wider (fewer wraps)
)
instructions.draw()
win.flip()


# Wait for any key press to continue
event.waitKeys()

# Clear screen
win.flip();


##trial structure:
# Show fixation cross and wait for ISI
fixation = visual.TextStim(win, text='+', color='white', height=50)

# Audio device indices (adjust these)
headphones_device = 4  # Device index for headphones
speakers_device = 5  # Device index for speakers


#repeat for x trials. each new trial should be on a new row in the results table
# Run trials in 3 blocks with breaks
practice_trials = 5

# Generate balanced trial list
number_of_trials = 96 #keep this at 96 for full experiment
number_of_blocks = 3
trials_per_block_count = number_of_trials // number_of_blocks
trial_list = []

# Trial list format: [output, stim_type]
# output: 0 = speakers, 1 = headphones
# stim_type: 0 = noise, 1 = ISTS, 2 = environment
for output in [0, 1]:  # 0 = speakers, 1 = headphones
    for stim_type in [0, 1, 2]:  # 0 = noise, 1 = ISTS, 2 = environment
        trials_per_combination = number_of_trials // 6  # 18 / 6 = 3
        for _ in range(trials_per_combination):
            trial_list.append([output, stim_type])

# Shuffle the trial list
random.shuffle(trial_list)

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
        gain_linear = 10 ** (gain_db / 20)
        audio_5s = audio_5s * gain_linear
        audio_5s = apply_safety_limit(audio_5s)
        if audio_5s.ndim > 1:
            audio_5s = audio_5s.mean(axis=1)

        sd.default.device = (None, device)
        print(f"Practice {p+1}/{practice_trials}: {playback_type} via device {device} ({stim_category})")

        # Show only fixation cross initially
        fixation.draw()
        win.flip()

        # START AUDIO PLAYBACK (non-blocking) and start timing immediately
        sd.play(audio_5s, samplerate=fs, device=device)
        start_time = time.time()
        
        # Wait 3 seconds, then show image below fixation cross
        image_shown = False
        response = None
        
        while response is None:
            elapsed_time = time.time() - start_time
            
            # Check if 3 seconds has passed and image hasn't been shown yet
            if not image_shown and elapsed_time >= 3.0:
                fixation.draw()
                if info_image:
                    info_image.draw()
                win.flip()
                image_shown = True
            
            # Only check for responses AFTER image is shown (after 3 seconds)
            if image_shown:
                keys = event.getKeys(keyList=['up', 'down', 'escape'], timeStamped=False)
                if keys:
                    rt = time.time() - start_time
                    if 'escape' in keys:
                        sd.stop()
                        win.close()
                        core.quit()
                    response = keys[0]
                    break
            else:
                # Clear any key presses before 3 seconds (ignore them)
                event.getKeys()
            
            core.wait(0.01)  # Small delay to prevent CPU overload
        
        # Stop audio after response
        sd.stop()
        
    except Exception as e:
        print(f"Error playing practice stimulus {stimulus}: {e}")
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
            gain_linear = 10 ** (gain_db / 20)
            audio_5s = audio_5s * gain_linear
            
            # Apply safety limiter to prevent dangerously loud playback
            audio_5s = apply_safety_limit(audio_5s)

            # Explicitly set output device (leave input as default)
            sd.default.device = (None, device)

            # Debug: log routing
            print(f"Trial {trial_index+1}/{number_of_trials}: {playback_type} via device {device} ({stim_category})")
            
            # Show only fixation cross initially
            fixation.draw()
            win.flip()
            
            # Start audio playback (non-blocking) and start timing immediately
            sd.play(audio_5s, samplerate=fs, device=device)
            start_time = time.time()
            
            # Wait 3 seconds, then show image below fixation cross
            image_shown = False
            response = None
            
            while response is None:
                elapsed_time = time.time() - start_time
                
                # Check if 3 seconds has passed and image hasn't been shown yet
                if not image_shown and elapsed_time >= 3.0:
                    fixation.draw()
                    if info_image:
                        info_image.draw()
                    win.flip()
                    image_shown = True
                
                # Only check for responses AFTER image is shown (after 3 seconds)
                if image_shown:
                    keys = event.getKeys(keyList=['up', 'down', 'escape'], timeStamped=False)
                    if keys:
                        rt = time.time() - start_time
                        if 'escape' in keys:
                            sd.stop()
                            win.close()
                            core.quit()
                        response = keys[0]
                        break
                else:
                    # Clear any key presses before 3 seconds (ignore them)
                    event.getKeys()
                
                core.wait(0.01)  # Small delay to prevent CPU overload
            
            # Stop audio after response
            sd.stop()
            
        except Exception as e:
            print(f"Error playing stimulus {stimulus}: {e}")
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

# Close window and quit
win.close()
core.quit()
