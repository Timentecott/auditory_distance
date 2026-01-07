from psychopy import visual, event, core
import pandas as pd
import time
import numpy as np
import random
import sounddevice as sd
import soundfile as sf
import os, glob


# Safety limiter for dangerous sound levels
MAX_SPL_DB = 85  # Maximum SPL in dB (85 dB is OSHA safe limit for 8 hours)
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
        print(f"  ⚠ Safety limiter engaged: reduced gain by {20 * np.log10(scaling_factor):.1f} dB")
    return audio


#load headphone stimuli from /localised_stimuli
base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
headphone_dir = os.path.join(base_dir, 'localised_stimuli')
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


#interstimulus interval
ISI = 1.0 #seconds

# results table: one row per trial
results = pd.DataFrame(columns=[
    'presentation_type',    # 'headphone' or 'speaker'
    'stimulus',        # stimulus filename or ID
    'stimulus_category', # environment, ISTS, or noise
    'response',         # key pressed by participant
    'rt',               # response time in seconds
    'accuracy',         # 1 = correct, 0 = incorrect
    'timestamp'         # trial timestamp
])

# Ensure results directory exists
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

def append_result(presentation_type, stimulus, response, rt, accuracy, stimulus_category=None):
    """Append one trial's data to the results table and persist to CSV immediately."""
    results.loc[len(results)] = {
        'presentation_type': presentation_type,
        'stimulus': stimulus,
        'stimulus_category': stimulus_category,
        'response': response,
        'rt': rt,
        'accuracy': accuracy,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    # Persist after each trial so practice trials are saved
    pid = globals().get('participant_id')
    fname = os.path.join(results_dir, f"{pid}.csv" if pid else "temp_results.csv")
    try:
        print(f"Saving results to {fname} (rows={len(results)})")
        results.to_csv(fname, index=False)
    except Exception as e:
        print(f"Warning: failed to save results to {fname}: {e}")

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
 
#display instructions: 

instructions_text = """You will hear sounds either through headphones or loudspeakers.

Your task is to identify whether the sound is played through headphones or loudspeakers.

Press the UP ARROW key for loudspeakers and the DOWN ARROW key for headphones.

Try to respond as quickly and accurately as possible.

Press any key to begin."""

instructions = visual.TextStim(win, text=instructions_text, color='white', height=30)
instructions.draw()
win.flip()


# Wait for any key press to continue
event.waitKeys()

# Clear screen
win.flip()


##trial structure:
# Show fixation cross and wait for ISI
fixation = visual.TextStim(win, text='+', color='white', height=50)
fixation.draw()
win.flip()
core.wait(ISI)  # Wait for interstimulus interval

#play a headphone or speaker stimulus at random. Record decision 
# if headphone, play via headphones, if speaker, play via speakers

# Audio device indices (adjust these for your setup)
headphones_device = 13  # Device index for headphones
speakers_device = 11  # Device index for speakers


#repeat for x trials. each new trial should be on a new row in the results table
# Run trials in 3 blocks with breaks
practice_trials = 3
block1trials = 4
block2trials = 4
block3trials = 4
total_trials = block1trials + block2trials + block3trials

# Create randomized trial order:
# 1. Half headphone, half speaker
trial_types = ['headphone'] * (total_trials // 2) + ['speaker'] * (total_trials // 2)
random.shuffle(trial_types)

# 2. Equal distribution of stimulus types (environment, ISTS, noise)
stim_types_per_trial = [stim_type for stim_type in stimulus_types for _ in range(total_trials // len(stimulus_types))]
random.shuffle(stim_types_per_trial)

# 3. Combine into trial list: (playback_type, stimulus_type)
trial_list = list(zip(trial_types, stim_types_per_trial))

print(f"\nTrial configuration:")
print(f"  Total trials: {total_trials}")
print(f"  Headphone trials: {sum(1 for t in trial_types if t == 'headphone')}")
print(f"  Speaker trials: {sum(1 for t in trial_types if t == 'speaker')}")
for stim_type in stimulus_types:
    print(f"  {stim_type.capitalize()} trials: {sum(1 for t in stim_types_per_trial if t == stim_type)}")
print()

trials_per_block = [block1trials, block2trials, block3trials]
trial_counter = 0

# --- Practice trials ---
# Define response_prompt and img_path before practice trials
response_prompt = visual.TextStim(
    win,
    text="Headphone or Speaker?\n\nUP ARROW for Speaker\nDOWN ARROW for Headphone",
    color='white',
    height=30,
    pos=(0, 150)
)
img_path = os.path.join(base_dir, 'resources', 'headphonevsloudspeak_info_graphic.png')

for p in range(practice_trials):
    trial_num = trial_counter

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
        gain_db = random.uniform(-10, 10)
        gain_linear = 10 ** (gain_db / 20)
        audio_5s = audio_5s * gain_linear
        audio_5s = apply_safety_limit(audio_5s)
        if audio_5s.ndim > 1:
            audio_5s = audio_5s.mean(axis=1)

        sd.default.device = (None, device)
        print(f"Practice {trial_num}: {playback_type} via device {device} ({stim_category})")

        fixation.draw()
        win.flip()

        sd.play(audio_5s, samplerate=fs, device=device)
        sd.wait()

        # Create info_image in practice loop to avoid NameError
        if os.path.exists(img_path):
            info_image = visual.ImageStim(
                win,
                image=img_path,
                pos=(0, -100),
                size=(800, 400)
            )
        else:
            info_image = None

        response_prompt.draw()
        if info_image:
            info_image.draw()
        win.flip()

        start_time = time.time()
        keys = event.waitKeys(keyList=['up', 'down', 'escape'])
        rt = time.time() - start_time
        if 'escape' in keys:
            win.close()
            core.quit()
        response = keys[0]
    except Exception as e:
        print(f"Error playing practice stimulus {stimulus}: {e}")
        trial_counter += 1
        continue

    if (response == 'up' and playback_type == 'speaker') or (response == 'down' and playback_type == 'headphone'):
        accuracy = 1
    else:
        accuracy = 0

    # Save practice trial with presentation_type indicating device + '(practice)'
    if playback_type == 'speaker':
        presentation_label = 'loudspeaker(practice)'
    else:
        presentation_label = 'headphone(practice)'
    append_result(presentation_label, stimulus, response, rt, accuracy, stimulus_category=stim_category)
    trial_counter += 1

# --- End practice trials ---

for block in range(3):
    for i in range(trials_per_block[block]):
        trial_num = trial_counter
        
        fixation.draw()
        win.flip()
        core.wait(ISI)
        
        # Get trial configuration: (playback_type, stimulus_type)
        playback_type, stim_category = trial_list[trial_counter]
        trial_counter += 1
        
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
            # Apply random gain between -10 and +10 dB
            gain_db = random.uniform(-10 , 10)
            gain_linear = 10 ** (gain_db / 20)
            audio_5s = audio_5s * gain_linear
            
            # Apply safety limiter to prevent dangerously loud playback
            audio_5s = apply_safety_limit(audio_5s)

            # Force mono to avoid channel/device mismatches
            if audio_5s.ndim > 1:
                audio_5s = audio_5s.mean(axis=1)

            # Explicitly set output device (leave input as default)
            sd.default.device = (None, device)

            # Debug: log routing
            print(f"Trial {trial_num}: {playback_type} via device {device} ({stim_category})")
            
            # Show fixation cross during audio playback
            fixation.draw()
            win.flip()
            
            # Play audio and wait for completion
            sd.play(audio_5s, samplerate=fs, device=device)
            playback_duration = len(audio_5s) / fs
            sd.wait()  # Block until audio finishes playing
            
            # After audio finishes, display response prompt with image
            response_prompt.draw()
            if os.path.exists(img_path):
                info_image = visual.ImageStim(
                    win,
                    image=img_path,
                    pos=(0, -100),
                    size=(800, 400)
                )
                info_image.draw()
            win.flip()
            
            # Start timing for response collection (after audio finished)
            start_time = time.time()
            keys = event.waitKeys(keyList=['up', 'down', 'escape'])
            rt = time.time() - start_time
            
            if 'escape' in keys:
                win.close()
                core.quit()
            
            response = keys[0]
        except Exception as e:
            print(f"Error playing stimulus {stimulus}: {e}")
            continue  # Skip this trial if error occurs
        
        if (response == 'up' and playback_type == 'speaker') or (response == 'down' and playback_type == 'headphone'):
            accuracy = 1
        else:
            accuracy = 0
        
        append_result(playback_type, stimulus, response, rt, accuracy, stimulus_category=stim_category)
    
    # Display break message after each block (except the last)
    if block < 2:
        if block == 0:
            break_msg = visual.TextStim(win, text="You're 1/3 of the way through.\n\nTake a break.\n\nPress any key to continue.", color='white', height=30)
        else:
            break_msg = visual.TextStim(win, text="You're 2/3 of the way through.\n\nTake a break.\n\nPress any key to continue.", color='white', height=30)
        break_msg.draw()
        win.flip()
        event.waitKeys()

#at end of experiment, save results table as a csv file wigth participant ID and timestamp in filename
#display "thank you for participating!"




# Close the window
win.close()
print(results)
#save results to csv with in results folder participant ID as filename
results_filename = f"results/{participant_id}.csv"
results.to_csv(results_filename, index=False)   
print(f"Results saved to {results_filename}")
core.quit()
