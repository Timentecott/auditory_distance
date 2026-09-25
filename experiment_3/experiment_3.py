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

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Audio setup
stimuli_dir = os.path.join(os.path.dirname(__file__), 'stimuli')

# Preload audio files for different presentation types
# Supports: in_situ, ex_situ, loudspeaker (with near/far for loudspeaker)
audio_files = {
    ('in_situ', 'near'): os.path.join(stimuli_dir, 'in_situ_near.wav'),
    ('in_situ', 'far'): os.path.join(stimuli_dir, 'in_situ_far.wav'),
    ('ex_situ', 'near'): os.path.join(stimuli_dir, 'ex_situ_near.wav'),
    ('ex_situ', 'far'): os.path.join(stimuli_dir, 'ex_situ_far.wav'),
    ('loudspeaker', 'near'): os.path.join(stimuli_dir, 'loudspeaker_near.wav'),
    ('loudspeaker', 'far'): os.path.join(stimuli_dir, 'loudspeaker_far.wav'),
}

# Load all audio files
audio_data = {}
fs = None

for (presentation_type, location), audio_file in audio_files.items():
    if os.path.exists(audio_file):
        audio, sr = sf.read(audio_file, dtype='float32')
        # Ensure stereo for headphone playback
        if presentation_type in ['in_situ', 'ex_situ']:
            audio = ensure_stereo(audio)
        audio_data[(presentation_type, location)] = np.asarray(audio, dtype=np.float32)
        if fs is None:
            fs = sr
        elif fs != sr:
            print(f"Warning: Sample rate mismatch for {audio_file}: expected {fs}Hz, got {sr}Hz")
    else:
        print(f"Warning: Audio file not found: {audio_file}")

# Fallback to single brown_noise file if specific files don't exist
if not audio_data:
    print("Loading fallback brown_noise_5s.wav...")
    fallback_file = os.path.join(stimuli_dir, 'brown_noise_5s.wav')
    if os.path.exists(fallback_file):
        audio, fs = sf.read(fallback_file, dtype='float32')
        audio_stereo = ensure_stereo(audio)
        # Use same audio for all presentation types
        for (presentation_type, location) in audio_files.keys():
            audio_data[(presentation_type, location)] = audio_stereo
    else:
        raise FileNotFoundError(f"No audio files found in {stimuli_dir}")

# Trial configuration for testing: 1 practice and 1 main trial
# Both with correct answer = 3
# question_type: 'congruent', 'incongruent', or 'outlier'
# variable_type: 'loudness', 'quietness', 'closeness', 'farness'
# presentation_type: 'in_situ', 'ex_situ', or 'loudspeaker'
# sound_location: 'near' or 'far'
trial_list = [
    # (trial_number, trial_type, question_type, variable_type, correct_answer, presentation_type, sound_location)
    (1, 'practice', 'outlier', 'farness', 3, 'loudspeaker', 'far'),
    (2, 'main', 'congruent', 'farness', 3, 'in_situ', 'near'),
]

results_data = []
all_sounds_heard = False
trial_playback_state = {
    'audio': None, 
    'pos': 0,
    'presentation_type': 'in_situ',
    'sound_location': 'near'
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
        presentation_type = trial_playback_state.get('presentation_type', 'in_situ')
        sound_location = trial_playback_state.get('sound_location', 'near')
        end_pos = pos + frame_count

        if end_pos <= audio.shape[0]:
            # Route audio through ASIO channels based on presentation type
            audio_frame = audio[pos:end_pos]
            routed = route_to_asio_channels(audio_frame, presentation_type, sound_location)
            outdata[:] = routed
            trial_playback_state['pos'] = end_pos
        else:
            remaining = audio.shape[0] - pos
            if remaining > 0:
                audio_frame = audio[pos:]
                routed = route_to_asio_channels(audio_frame, presentation_type, sound_location)
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
         "Click the buttons to hear sounds.\n"
         "After listening to all sounds, press the corresponding number (1, 2, or 3)\n"
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
         "For the following trials, click the buttons to hear sounds.\n"
         "After listening to all sounds, press the corresponding number (1, 2, or 3)\n"
         "to indicate your answer.\n\n"
         "No feedback will be provided.",
    color='white',
    height=30,
    wrapWidth=1200
)

# Run trials
for trial_num, trial_type, question_type, variable_type, correct_answer, presentation_type, sound_location in trial_list:
    if trial_num == 2:
        # Show main instructions before first main trial
        main_instructions.draw()
        win.flip()
        event.waitKeys()

    sounds_heard = set()  # Track which sounds (1, 2, 3) have been played
    trial_response = None
    trial_rt = None
    trial_start_time = None
    warning_start_time = None
    showing_warning = False
    selected_option = None  # Track selected option: 'A', 'B', or 'C'
    follow_up_response = None  # Track follow-up question response
    difference_options = ['Quieter', 'Louder', 'Closer', 'Further', 'More Realistic', 'Less Realistic']

    while trial_response is None:
        # Determine instruction text based on question_type and variable_type
        if question_type == 'outlier':
            instruction_text = "Which is the odd one out?"
        elif question_type == 'congruent':
            # Question matches the variable
            if variable_type == 'loudness':
                instruction_text = "Which is the loudest?"
            elif variable_type == 'quietness':
                instruction_text = "Which is the quietest?"
            elif variable_type == 'closeness':
                instruction_text = "Which is the closest?"
            elif variable_type == 'farness':
                instruction_text = "Which is the furthest?"
        elif question_type == 'incongruent':
            # Question doesn't match the variable
            if variable_type == 'closeness':
                instruction_text = "Which is the loudest?"
            elif variable_type == 'farness':
                instruction_text = "Which is the quietest?"
            elif variable_type == 'loudness':
                instruction_text = "Which is the closest?"
            elif variable_type == 'quietness':
                instruction_text = "Which is the furthest?"

        # Create audio playback buttons (numbered 1, 2, 3)
        # Positioned to be visible and well-centered
        button_1 = visual.Rect(
            win,
            width=120,
            height=120,
            pos=(-200, 180),
            fillColor=[0.2, 0.4, 0.6],  # Nice blue
            lineColor='white',
            lineWidth=2
        )
        button_2 = visual.Rect(
            win,
            width=120,
            height=120,
            pos=(0, 180),
            fillColor=[0.2, 0.4, 0.6],
            lineColor='white',
            lineWidth=2
        )
        button_3 = visual.Rect(
            win,
            width=120,
            height=120,
            pos=(200, 180),
            fillColor=[0.2, 0.4, 0.6],
            lineColor='white',
            lineWidth=2
        )

        label_1 = visual.TextStim(win, text='1', color='white', height=50, pos=(-200, 180), bold=True)
        label_2 = visual.TextStim(win, text='2', color='white', height=50, pos=(0, 180), bold=True)
        label_3 = visual.TextStim(win, text='3', color='white', height=50, pos=(200, 180), bold=True)

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

        # Create response selection boxes aligned with top buttons
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

        prompt_text = "Click to hear sounds"
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

        # Check for mouse clicks on buttons
        mouse = event.Mouse()

        # Detect mouse position for hover effects
        mouse_pos = mouse.getPos()

        # Draw everything in order
        # Draw containers first
        play_button_section.draw()
        question_section.draw()
        response_section.draw()

        # Draw text/buttons on top
        prompt.draw()

        # Draw play buttons with hover effect
        button_1_hover = mouse.isPressedIn(button_1) or (abs(mouse_pos[0] - (-200)) < 60 and abs(mouse_pos[1] - 350) < 60)
        button_2_hover = mouse.isPressedIn(button_2) or (abs(mouse_pos[0] - 0) < 60 and abs(mouse_pos[1] - 350) < 60)
        button_3_hover = mouse.isPressedIn(button_3) or (abs(mouse_pos[0] - 200) < 60 and abs(mouse_pos[1] - 350) < 60)

        button_1.fillColor = [0.25, 0.5, 0.7] if button_1_hover else [0.2, 0.4, 0.6]
        button_2.fillColor = [0.25, 0.5, 0.7] if button_2_hover else [0.2, 0.4, 0.6]
        button_3.fillColor = [0.25, 0.5, 0.7] if button_3_hover else [0.2, 0.4, 0.6]

        button_1.draw()
        button_2.draw()
        button_3.draw()
        label_1.draw()
        label_2.draw()
        label_3.draw()

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

        # Check for mouse clicks on audio playback buttons
        if mouse.isPressedIn(button_1):
            with trial_state_lock:
                audio_key = (presentation_type, sound_location)
                if audio_key in audio_data:
                    trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                    trial_playback_state['pos'] = 0
                    trial_playback_state['presentation_type'] = presentation_type
                    trial_playback_state['sound_location'] = sound_location
            sounds_heard.add(1)
        elif mouse.isPressedIn(button_2):
            with trial_state_lock:
                audio_key = (presentation_type, sound_location)
                if audio_key in audio_data:
                    trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                    trial_playback_state['pos'] = 0
                    trial_playback_state['presentation_type'] = presentation_type
                    trial_playback_state['sound_location'] = sound_location
            sounds_heard.add(2)
        elif mouse.isPressedIn(button_3):
            with trial_state_lock:
                audio_key = (presentation_type, sound_location)
                if audio_key in audio_data:
                    trial_playback_state['audio'] = audio_data[audio_key].astype(np.float32)
                    trial_playback_state['pos'] = 0
                    trial_playback_state['presentation_type'] = presentation_type
                    trial_playback_state['sound_location'] = sound_location
            sounds_heard.add(3)

        # Check for mouse clicks on response selection boxes
        elif mouse.isPressedIn(response_box_a):
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

        # Create option boxes in two rows (3 per row)
        mouse = event.Mouse()
        option_positions = [
            (-250, 150), (0, 150), (250, 150),  # Top row
            (-250, 50), (0, 50), (250, 50)      # Bottom row
        ]

        option_boxes = []
        option_labels = []

        for i, (pos, text) in enumerate(zip(option_positions, difference_options)):
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
        for box in option_boxes:
            box.draw()
        for label in option_labels:
            label.draw()

        # Check for mouse clicks on options
        for i, box in enumerate(option_boxes):
            if mouse.isPressedIn(box):
                follow_up_response = difference_options[i]
                selected_difference = i

        win.flip()
        core.wait(0.05)

    # Record trial results
    results_data.append({
        'participant_id': participant_num,
        'trial_number': trial_num,
        'trial_type': trial_type,
        'question_type': question_type,
        'variable_type': variable_type,
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
