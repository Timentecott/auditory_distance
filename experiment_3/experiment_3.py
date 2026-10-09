#3AFC questions
#are there any conditions where participants actually can't tell a difference between headphone and loudspeaker? ie .33 chance of correct answer 

#2afc questions
#which one is further/closer? these questions are equivilant. At what point do participants perceive in/ex situ audio as further closer? Is there an effect of asking further/closer compared to louder quieter?
#do participants ever choose a louder stimulus as closer or quieter stimulus as further? 

#we could just do 3AFC and then ask participants why they chose it as the odd one out? 




import os
os.environ["SD_ENABLE_ASIO"] = "1" #this line is important as it allows revelation of asio devices
import pandas as pd
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
from psychopy import visual, event, core
from pathlib import Path


def ensure_stereo(audio):
    """Force audio to stereo (L/R) for playback."""
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

#initialise changeable things 
index_to_asio = 12

# Initialize PsychoPy window
win = visual.Window(size=(1920, 1080), color='black', fullscr=True, units='pix')

# Get participant number
dlg_text = visual.TextStim(
    win,
    text="Enter participant number:",
    color='white',
    height=40
)
dlg_text.draw()
win.flip()

participant_num = ""
while True:
    keys = event.getKeys()
    for key in keys:
        if key == 'return':
            if participant_num:
                break
        elif key == 'backspace':
            participant_num = participant_num[:-1]
        elif key.isdigit():
            participant_num += key

    dlg_text.draw()
    input_text = visual.TextStim(
        win,
        text=participant_num,
        color='white',
        height=40,
        pos=(0, -60)
    )
    input_text.draw()
    win.flip()

    if 'return' in keys and participant_num:
        break
    core.wait(0.05)

# Set random seed based on participant number for reproducibility
np.random.seed(int(participant_num))

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Audio setup
stimuli_dir = os.path.join(os.path.dirname(__file__), 'stimuli')
experiment_dir = os.path.dirname(__file__)

# Load conditions from CSV
conditions_file = os.path.join(experiment_dir, 'conditions.csv')
if not os.path.exists(conditions_file):
    raise FileNotFoundError(f"Conditions file not found: {conditions_file}")

conditions_df = pd.read_csv(conditions_file)

# Load all audio files referenced in the conditions
audio_data = {}
fs = None

for idx, row in conditions_df.iterrows():
    odd_file_path = os.path.join(experiment_dir, row['odd_file'])
    control_file_path = os.path.join(experiment_dir, row['control_file'])

    # Load odd file
    odd_key = f"trial_{row['trial_number']}_odd"
    if os.path.exists(odd_file_path):
        audio, sr = sf.read(odd_file_path, dtype='float32')
        audio = ensure_stereo(audio)
        audio_data[odd_key] = np.asarray(audio, dtype=np.float32)
        if fs is None:
            fs = sr
    else:
        print(f"Warning: Odd audio file not found: {odd_file_path}")

    # Load control file
    control_key = f"trial_{row['trial_number']}_control"
    if os.path.exists(control_file_path):
        audio, sr = sf.read(control_file_path, dtype='float32')
        audio = ensure_stereo(audio)
        audio_data[control_key] = np.asarray(audio, dtype=np.float32)
        if fs is None:
            fs = sr
    else:
        print(f"Warning: Control audio file not found: {control_file_path}")

if not audio_data:
    raise FileNotFoundError(f"No audio files loaded from conditions")

# Convert conditions CSV to trial_list format
trial_list = []
for idx, row in conditions_df.iterrows():
    trial_list.append((
        int(row['trial_number']),
        row['trial_type'],
        row['condition'],
        row['playback_type']
    ))

results_data = []
all_sounds_heard = False

def get_odd_position_for_trial(trial_number):
    """Determine the odd sound position (1, 2, or 3) for a trial using seeded RNG.

    Args:
        trial_number: The trial number (used to generate unique random position per trial)

    Returns:
        Odd sound position (1, 2, or 3)
    """
    # Create a fresh random state for this trial to ensure reproducibility
    # while allowing different positions for different trials
    rng = np.random.RandomState(int(participant_num) + trial_number)
    return rng.randint(1, 4)

def route_playback(audio, playback_type, is_odd_sound):
    """Route audio to ASIO channels based on playback_type.

    Args:
        audio: Audio array (stereo)
        playback_type: 'Always headphone', 'Always loudspeaker', 'Mix (odd headphone)', 'Mix (odd loudspeaker)'
        is_odd_sound: Boolean indicating if this is the odd sound

    Returns:
        4-channel ASIO routed audio
    """
    routed = np.zeros((audio.shape[0], 4), dtype=np.float32)
    stereo_audio = ensure_stereo(audio)
    mono_audio = audio[:, 0] if audio.ndim > 1 else audio

    # Normalize playback_type for case-insensitive comparison
    playback_type_lower = playback_type.lower()

    if playback_type_lower == 'always headphone':
        # All sounds to headphone (channels 0-1)
        routed[:, 0:2] = stereo_audio[:, :2]
    elif playback_type_lower == 'always loudspeaker':
        # All sounds to loudspeaker (channel 2)
        routed[:, 2] = mono_audio
    elif playback_type_lower == 'mix (odd headphone)':
        # Odd to headphone, others to loudspeaker
        if is_odd_sound:
            routed[:, 0:2] = stereo_audio[:, :2]
        else:
            routed[:, 2] = mono_audio
    elif playback_type_lower == 'mix (odd loudspeaker)':
        # Odd to loudspeaker, others to headphone
        if is_odd_sound:
            routed[:, 2] = mono_audio
        else:
            routed[:, 0:2] = stereo_audio[:, :2]
    else:
        raise ValueError(f"Unknown playback_type: {playback_type}")

    return routed

trial_playback_state = {
    'audio': None, 
    'pos': 0,
    'playback_type': 'Always loudspeaker',
    'is_odd_sound': False
}
trial_state_lock = threading.Lock()


def trial_playback_callback(outdata, frame_count, time_info, status):
    """Callback for streaming audio during trial."""
    if status:
        print(f"Audio status: {status}")

    with trial_state_lock:
        if trial_playback_state['audio'] is None:
            outdata.fill(0)
            return

        audio = trial_playback_state['audio']
        pos = trial_playback_state['pos']
        playback_type = trial_playback_state.get('playback_type', 'Always loudspeaker')
        is_odd_sound = trial_playback_state.get('is_odd_sound', False)
        end_pos = pos + frame_count

        if end_pos <= audio.shape[0]:
            # Route audio through ASIO channels based on playback type
            audio_frame = audio[pos:end_pos]
            routed = route_playback(audio_frame, playback_type, is_odd_sound)
            outdata[:] = routed
            trial_playback_state['pos'] = end_pos
        else:
            remaining = audio.shape[0] - pos
            if remaining > 0:
                audio_frame = audio[pos:]
                routed = route_playback(audio_frame, playback_type, is_odd_sound)
                # Pad with silence to match frame_count
                padding = np.zeros((frame_count - remaining, 4), dtype=np.float32)
                outdata[:remaining] = routed
                outdata[remaining:] = padding
            else:
                outdata.fill(0)
            trial_playback_state['pos'] = audio.shape[0]


# Open persistent audio stream with ASIO device (4 channels for headphone/speaker routing)
stream = sd.OutputStream(
    samplerate=fs,
    device=index_to_asio,  # Use ASIO device for multi-channel routing
    channels=4,
    dtype='float32',
    callback=trial_playback_callback,
)
stream.start()

# Practice instructions
practice_instructions = visual.TextStim(
    win,
    text="PRACTICE TRIAL\n\n"
         "Press 1, 2, or 3 to hear the sounds.\n"
         "After listening to all sounds, click the boxes below\n"
         "to indicate which sound is the odd one out.\n\n"
         "You will receive feedback on your answer.",
    color='white',
    height=30,
    wrapWidth=1200
)
practice_instructions.draw()
win.flip()
event.waitKeys()

# Main instructions
main_instructions = visual.TextStim(
    win,
    text="MAIN TRIALS\n\n"
         "Press 1, 2, or 3 to hear the sounds.\n"
         "After listening to all sounds, click the boxes below\n"
         "to indicate your answer.\n\n"
         "No feedback will be provided.",
    color='white',
    height=30,
    wrapWidth=1200
)

# Run trials
for trial_num, trial_type, condition, playback_type in trial_list:
    if trial_num == 2:
        # Show main instructions before first main trial
        main_instructions.draw()
        win.flip()
        event.waitKeys()

    # Determine odd sound position for this trial using seeded RNG
    correct_answer = get_odd_position_for_trial(trial_num)

    # Get the audio duration for progress bar
    odd_key = f"trial_{trial_num}_odd"
    audio_duration = audio_data[odd_key].shape[0] / fs if odd_key in audio_data else 5.0

    sounds_heard = set()  # Track which sounds (1, 2, 3) have been played
    trial_response = None
    trial_rt = None
    trial_start_time = None
    warning_start_time = None
    showing_warning = False
    selected_option = None  # Track selected option: 'A', 'B', or 'C'
    follow_up_response = None  # Track follow-up question response
    # Note: follow-up options are separate and only have Closer/Further

    while trial_response is None:
        # Question is always "Which is the odd one out?"
        instruction_text = "Which is the odd one out?"

        # Progress bar for audio playback
        playback_progress = 0.0
        if trial_playback_state['audio'] is not None:
            playback_progress = min(1.0, trial_playback_state['pos'] / trial_playback_state['audio'].shape[0])

        # Create keyboard press instructions
        keyboard_instruction = visual.TextStim(
            win,
            text="Press 1, 2, or 3 to hear sounds",
            color=[0.7, 0.7, 0.7],
            height=25,
            pos=(0, 250),
            wrapWidth=1200
        )

        # Progress bar visualization
        progress_bar_bg = visual.Rect(
            win,
            width=600,
            height=20,
            pos=(0, 200),
            fillColor=[0.15, 0.15, 0.15],
            lineColor=[0.4, 0.4, 0.4],
            lineWidth=1
        )

        progress_bar_fill = visual.Rect(
            win,
            width=600 * playback_progress,
            height=20,
            pos=(-300 + 300 * playback_progress, 200),
            fillColor=[0.3, 0.6, 0.9],
            lineColor=[0.3, 0.6, 0.9],
            lineWidth=0
        )

        # Sound labels (1, 2, 3) - not clickable, just labels
        label_1 = visual.TextStim(win, text='1', color='white', height=30, pos=(-200, 100), bold=True)
        label_2 = visual.TextStim(win, text='2', color='white', height=30, pos=(0, 100), bold=True)
        label_3 = visual.TextStim(win, text='3', color='white', height=30, pos=(200, 100), bold=True)

        # Show which sounds have been heard
        status_1 = visual.TextStim(win, text='✓' if 1 in sounds_heard else '○', color='green' if 1 in sounds_heard else [0.5, 0.5, 0.5], height=20, pos=(-200, 60), bold=True)
        status_2 = visual.TextStim(win, text='✓' if 2 in sounds_heard else '○', color='green' if 2 in sounds_heard else [0.5, 0.5, 0.5], height=20, pos=(0, 60), bold=True)
        status_3 = visual.TextStim(win, text='✓' if 3 in sounds_heard else '○', color='green' if 3 in sounds_heard else [0.5, 0.5, 0.5], height=20, pos=(200, 60), bold=True)

        # Question/instruction - large and prominent, centered
        instruction = visual.TextStim(
            win,
            text=instruction_text,
            color='white',
            height=50,
            pos=(0, 70),
            wrapWidth=1200,
            bold=True
        )

        # Create response selection boxes aligned with buttons
        response_box_a = visual.Rect(
            win,
            width=100,
            height=100,
            pos=(-200, -50),
            fillColor=[0.15, 0.3, 0.45] if selected_option == 'A' else [0.25, 0.35, 0.5],
            lineColor='white' if selected_option == 'A' else [0.5, 0.5, 0.5],
            lineWidth=3 if selected_option == 'A' else 2
        )
        response_box_b = visual.Rect(
            win,
            width=100,
            height=100,
            pos=(0, -50),
            fillColor=[0.15, 0.3, 0.45] if selected_option == 'B' else [0.25, 0.35, 0.5],
            lineColor='white' if selected_option == 'B' else [0.5, 0.5, 0.5],
            lineWidth=3 if selected_option == 'B' else 2
        )
        response_box_c = visual.Rect(
            win,
            width=100,
            height=100,
            pos=(200, -50),
            fillColor=[0.15, 0.3, 0.45] if selected_option == 'C' else [0.25, 0.35, 0.5],
            lineColor='white' if selected_option == 'C' else [0.5, 0.5, 0.5],
            lineWidth=3 if selected_option == 'C' else 2
        )

        # 1, 2, 3 labels for response selection boxes
        label_a = visual.TextStim(win, text='1', color='white', height=40, pos=(-200, -50), bold=True)
        label_b = visual.TextStim(win, text='2', color='white', height=40, pos=(0, -50), bold=True)
        label_c = visual.TextStim(win, text='3', color='white', height=40, pos=(200, -50), bold=True)

        # Create confirm button
        all_sounds_heard = len(sounds_heard) == 3
        confirm_button_color = [0.1, 0.6, 0.2] if (all_sounds_heard and selected_option) else [0.3, 0.3, 0.3]
        confirm_button = visual.Rect(
            win,
            width=150,
            height=60,
            pos=(0, -170),
            fillColor=confirm_button_color,
            lineColor='white' if (all_sounds_heard and selected_option) else [0.5, 0.5, 0.5],
            lineWidth=2
        )
        confirm_label = visual.TextStim(
            win,
            text='Confirm',
            color='white' if (all_sounds_heard and selected_option) else [0.7, 0.7, 0.7],
            height=30,
            pos=(0, -170),
            bold=True
        )

        prompt_text = "Press keys 1, 2, or 3 to hear sounds"
        if not all_sounds_heard:
            prompt_text += f" ({len(sounds_heard)}/3 heard)"

        prompt = visual.TextStim(
            win,
            text=prompt_text,
            color=[0.9, 0.9, 0.9],
            height=20,
            pos=(0, 280),
            wrapWidth=1200
        )

        # Visual containers for better grouping
        # Container for play buttons section
        play_button_section = visual.Rect(
            win,
            width=700,
            height=200,
            pos=(0, 180),
            fillColor=[0.05, 0.05, 0.1],
            lineColor=[0.4, 0.4, 0.4],
            lineWidth=1
        )

        # Container for question section
        question_section = visual.Rect(
            win,
            width=800,
            height=120,
            pos=(0, 70),
            fillColor=[0.08, 0.08, 0.12],
            lineColor=[0.3, 0.3, 0.3],
            lineWidth=1
        )

        # Container for response buttons section
        response_section = visual.Rect(
            win,
            width=500,
            height=180,
            pos=(0, -50),
            fillColor=[0.05, 0.05, 0.1],
            lineColor=[0.4, 0.4, 0.4],
            lineWidth=1
        )

        error_text = visual.TextStim(
            win,
            text="Please listen to all sounds before making a choice.",
            color='red',
            height=40,
            pos=(0, 50),
            wrapWidth=1200
        )

        # Draw everything
        keyboard_instruction.draw()
        progress_bar_bg.draw()
        progress_bar_fill.draw()

        label_1.draw()
        label_2.draw()
        label_3.draw()

        status_1.draw()
        status_2.draw()
        status_3.draw()

        # Draw question
        instruction.draw()

        # Draw response selection boxes
        response_box_a.draw()
        response_box_b.draw()
        response_box_c.draw()
        label_a.draw()
        label_b.draw()
        label_c.draw()

        # Draw confirm button (enabled or disabled)
        confirm_button.draw()
        confirm_label.draw()

        if trial_start_time is None:
            trial_start_time = time.time()

        # Display warning if recently triggered
        if showing_warning:
            elapsed = time.time() - warning_start_time
            if elapsed < 5:
                error_text.draw()
            else:
                showing_warning = False
                warning_start_time = None

        # Check for keyboard presses (1, 2, 3) to play sounds
        keys = event.getKeys()
        for key in keys:
            if key == '1':
                with trial_state_lock:
                    is_odd = (1 == correct_answer)
                    audio_key = f"trial_{trial_num}_{'odd' if is_odd else 'control'}"
                    if audio_key in audio_data:
                        trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                        trial_playback_state['pos'] = 0
                        trial_playback_state['playback_type'] = playback_type
                        trial_playback_state['is_odd_sound'] = is_odd
                sounds_heard.add(1)
            elif key == '2':
                with trial_state_lock:
                    is_odd = (2 == correct_answer)
                    audio_key = f"trial_{trial_num}_{'odd' if is_odd else 'control'}"
                    if audio_key in audio_data:
                        trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                        trial_playback_state['pos'] = 0
                        trial_playback_state['playback_type'] = playback_type
                        trial_playback_state['is_odd_sound'] = is_odd
                sounds_heard.add(2)
            elif key == '3':
                with trial_state_lock:
                    is_odd = (3 == correct_answer)
                    audio_key = f"trial_{trial_num}_{'odd' if is_odd else 'control'}"
                    if audio_key in audio_data:
                        trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                        trial_playback_state['pos'] = 0
                        trial_playback_state['playback_type'] = playback_type
                        trial_playback_state['is_odd_sound'] = is_odd
                sounds_heard.add(3)

        # Check for mouse clicks on response selection boxes
        mouse = event.Mouse()
        if mouse.isPressedIn(response_box_a):
            selected_option = 'A'
        elif mouse.isPressedIn(response_box_b):
            selected_option = 'B'
        elif mouse.isPressedIn(response_box_c):
            selected_option = 'C'

        # Check for confirm button click (only if all sounds heard and option selected)
        elif mouse.isPressedIn(confirm_button) and all_sounds_heard and selected_option:
            # Convert A/B/C to 1/2/3
            response_map = {'A': 1, 'B': 2, 'C': 3}
            trial_response = response_map[selected_option]
            trial_rt = time.time() - trial_start_time
            # Stop audio playback
            with trial_state_lock:
                trial_playback_state['audio'] = None
                trial_playback_state['pos'] = 0

        win.flip()
        core.wait(0.05)

    # Check if answer is correct
    is_correct = 1 if trial_response == correct_answer else 0

    # Show feedback for practice trials
    if trial_type == 'practice':
        if is_correct:
            feedback = visual.TextStim(
                win,
                text="Correct!",
                color='green',
                height=50
            )
        else:
            feedback = visual.TextStim(
                win,
                text=f"Incorrect. The correct answer was {correct_answer}.",
                color='red',
                height=50
            )

        feedback.draw()
        win.flip()
        core.wait(2)

    # Follow-up question: Ask how the chosen sound was different
    follow_up_response = None
    selected_difference = None
    # For practice trials, add a third option "They're completely different!"
    if trial_type == 'practice':
        followup_difference_options = ['Closer', 'Further', "They're completely different!"]
    else:
        followup_difference_options = ['Closer', 'Further']

    while follow_up_response is None:
        # Create follow-up question text
        followup_question = visual.TextStim(
            win,
            text=f"In what way was sound {trial_response} different?\nIt was...",
            color='white',
            height=40,
            pos=(0, 300),
            wrapWidth=1200,
            bold=True
        )

        replay_instruction = visual.TextStim(
            win,
            text="Press 1, 2, or 3 to hear sounds again:",
            color=[0.9, 0.9, 0.9],
            height=20,
            pos=(0, 250),
            wrapWidth=1200
        )

        # Create option boxes
        mouse = event.Mouse()
        if trial_type == 'practice':
            option_positions = [
                (-200, 100), (0, 100), (200, 100)  # Three options for practice
            ]
        else:
            option_positions = [
                (-100, 100), (100, 100)  # Two options for main
            ]

        option_boxes = []
        option_labels = []

        for i, (pos, text) in enumerate(zip(option_positions, followup_difference_options)):
            box = visual.Rect(
                win,
                width=140,
                height=80,
                pos=pos,
                fillColor=[0.25, 0.35, 0.5],
                lineColor=[0.5, 0.5, 0.5],
                lineWidth=2
            )
            label = visual.TextStim(
                win,
                text=text,
                color='white',
                height=20,
                pos=pos,
                wrapWidth=130,
                bold=True
            )
            option_boxes.append(box)
            option_labels.append(label)

        # Draw follow-up question screen
        followup_question.draw()
        replay_instruction.draw()

        for box in option_boxes:
            box.draw()
        for label in option_labels:
            label.draw()

        # Check for keyboard presses (1, 2, 3) to replay sounds
        keys = event.getKeys()
        for key in keys:
            if key == '1':
                with trial_state_lock:
                    is_odd = (1 == correct_answer)
                    audio_key = f"trial_{trial_num}_{'odd' if is_odd else 'control'}"
                    if audio_key in audio_data:
                        trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                        trial_playback_state['pos'] = 0
                        trial_playback_state['playback_type'] = playback_type
                        trial_playback_state['is_odd_sound'] = is_odd
            elif key == '2':
                with trial_state_lock:
                    is_odd = (2 == correct_answer)
                    audio_key = f"trial_{trial_num}_{'odd' if is_odd else 'control'}"
                    if audio_key in audio_data:
                        trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                        trial_playback_state['pos'] = 0
                        trial_playback_state['playback_type'] = playback_type
                        trial_playback_state['is_odd_sound'] = is_odd
            elif key == '3':
                with trial_state_lock:
                    is_odd = (3 == correct_answer)
                    audio_key = f"trial_{trial_num}_{'odd' if is_odd else 'control'}"
                    if audio_key in audio_data:
                        trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                        trial_playback_state['pos'] = 0
                        trial_playback_state['playback_type'] = playback_type
                        trial_playback_state['is_odd_sound'] = is_odd

        # Check for mouse clicks on options
        if mouse.isPressedIn(option_boxes[0]):
            follow_up_response = followup_difference_options[0]
            selected_difference = 0
            # Stop audio playback
            with trial_state_lock:
                trial_playback_state['audio'] = None
                trial_playback_state['pos'] = 0
        elif mouse.isPressedIn(option_boxes[1]):
            follow_up_response = followup_difference_options[1]
            selected_difference = 1
            # Stop audio playback
            with trial_state_lock:
                trial_playback_state['audio'] = None
                trial_playback_state['pos'] = 0
        elif trial_type == 'practice' and len(option_boxes) > 2 and mouse.isPressedIn(option_boxes[2]):
            follow_up_response = followup_difference_options[2]
            selected_difference = 2
            # Stop audio playback
            with trial_state_lock:
                trial_playback_state['audio'] = None
                trial_playback_state['pos'] = 0

        win.flip()
        core.wait(0.05)

    # Record trial results
    results_data.append({
        'participant_id': participant_num,
        'trial_number': trial_num,
        'trial_type': trial_type,
        'condition': condition,
        'playback_type': playback_type,
        'correct_answer': correct_answer,
        'participant_response': trial_response,
        'response_time': trial_rt,
        'is_correct': is_correct,
        'follow_up_response': follow_up_response
    })

# Stop audio stream
stream.stop()
stream.close()

# Save results to CSV
results_df = pd.DataFrame(results_data)
results_file = os.path.join(results_dir, f'{participant_num}.csv')
results_df.to_csv(results_file, index=False)

print(f"Results saved to {results_file}")

# End screen
end_text = visual.TextStim(
    win,
    text="Experiment complete. Thank you!",
    color='white',
    height=40
)
end_text.draw()
win.flip()
core.wait(3)

win.close()
core.quit()
