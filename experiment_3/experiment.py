#3AFC questions
#are there any conditions where participants actually can't tell a difference between headphone and loudspeaker? ie .33 chance of correct answer 

#2afc questions
#which one is further/closer? these questions are equivilant. At what point do participants perceive in/ex situ audio as further closer? Is there an effect of asking further/closer compared to louder quieter?
#do participants ever choose a louder stimulus as closer or quieter stimulus as further? 


import os
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
audio_file = os.path.join(stimuli_dir, 'brown_noise_5s.wav')

# Load audio once
audio_data, fs = sf.read(audio_file)
audio_stereo = ensure_stereo(audio_data)

# Trial configuration for testing: 1 practice and 1 main trial
# Both with correct answer = 3
# question_type: 'congruent', 'incongruent', or 'outlier'
# variable_type: 'loudness', 'quietness', 'closeness', 'farness'
trial_list = [
    # (trial_number, trial_type, question_type, variable_type, correct_answer)
    (1, 'practice', 'outlier', 'farness', 3),
    (2, 'main', 'congruent', 'farness', 3),
]

results_data = []
all_sounds_heard = False
trial_playback_state = {'audio': None, 'pos': 0}
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
        end_pos = pos + frame_count

        if end_pos <= audio.shape[0]:
            outdata[:] = audio[pos:end_pos]
            trial_playback_state['pos'] = end_pos
        else:
            remaining = audio.shape[0] - pos
            if remaining > 0:
                outdata[:remaining] = audio[pos:]
                outdata[remaining:] = 0
            else:
                outdata.fill(0)
            trial_playback_state['pos'] = audio.shape[0]


# Open persistent audio stream with default device
stream = sd.OutputStream(
    samplerate=fs,
    device=None,  # Use default device
    channels=2,
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
for trial_num, trial_type, question_type, variable_type, correct_answer in trial_list:
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

        # Create buttons
        button_1 = visual.Rect(
            win,
            width=150,
            height=150,
            pos=(-250, 0),
            fillColor='gray',
            lineColor='white',
            lineWidth=3
        )
        button_2 = visual.Rect(
            win,
            width=150,
            height=150,
            pos=(0, 0),
            fillColor='gray',
            lineColor='white',
            lineWidth=3
        )
        button_3 = visual.Rect(
            win,
            width=150,
            height=150,
            pos=(250, 0),
            fillColor='gray',
            lineColor='white',
            lineWidth=3
        )

        label_1 = visual.TextStim(win, text='1', color='white', height=60, pos=(-250, 0))
        label_2 = visual.TextStim(win, text='2', color='white', height=60, pos=(0, 0))
        label_3 = visual.TextStim(win, text='3', color='white', height=60, pos=(250, 0))

        instruction = visual.TextStim(
            win,
            text=instruction_text,
            color='white',
            height=40,
            pos=(0, 350),
            wrapWidth=1200
        )

        all_sounds_heard = len(sounds_heard) == 3
        prompt_text = "Click buttons to hear sounds. Press 1, 2, or 3 to answer."
        if not all_sounds_heard:
            prompt_text += f"\n(Listen to all sounds first - heard {len(sounds_heard)}/3)"

        prompt = visual.TextStim(
            win,
            text=prompt_text,
            color='yellow' if not all_sounds_heard else 'white',
            height=25,
            pos=(0, -400),
            wrapWidth=1200
        )

        error_text = visual.TextStim(
            win,
            text="Please listen to all sounds before making a choice.",
            color='red',
            height=40,
            pos=(0, -250),
            wrapWidth=1200
        )

        # Check for mouse clicks on buttons
        mouse = event.Mouse()

        # Draw everything
        instruction.draw()
        button_1.draw()
        button_2.draw()
        button_3.draw()
        label_1.draw()
        label_2.draw()
        label_3.draw()
        prompt.draw()

        if trial_start_time is None:
            trial_start_time = time.time()

        # Check for keyboard response (1, 2, or 3)
        keys = event.getKeys()
        for key in keys:
            if key in ['1', '2', '3']:
                if not all_sounds_heard:
                    # Start showing warning for 5 seconds
                    showing_warning = True
                    warning_start_time = time.time()
                else:
                    trial_response = int(key)
                    trial_rt = time.time() - trial_start_time

        # Display warning if recently triggered
        if showing_warning:
            elapsed = time.time() - warning_start_time
            if elapsed < 5:
                error_text.draw()
            else:
                showing_warning = False
                warning_start_time = None

        # Check for mouse clicks
        if mouse.isPressedIn(button_1):
            with trial_state_lock:
                trial_playback_state['audio'] = audio_stereo.astype(np.float32)
                trial_playback_state['pos'] = 0
            sounds_heard.add(1)
        elif mouse.isPressedIn(button_2):
            with trial_state_lock:
                trial_playback_state['audio'] = audio_stereo.astype(np.float32)
                trial_playback_state['pos'] = 0
            sounds_heard.add(2)
        elif mouse.isPressedIn(button_3):
            with trial_state_lock:
                trial_playback_state['audio'] = audio_stereo.astype(np.float32)
                trial_playback_state['pos'] = 0
            sounds_heard.add(3)

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
        'is_correct': is_correct
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
