from psychopy import visual, event, core
from psychopy.hardware import keyboard
import os
import time
import pandas as pd
import sounddevice as sd
import soundfile as sf

os.environ["SD_ENABLE_ASIO"] = "1"

base_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(base_dir, "audio_stimuli", "Localised")
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

sound_prefix = "pink_noise_48k_30s_300_8000hz"
near_sound_path = os.path.join(audio_dir, f"{sound_prefix}_down.wav")
far_sound_path = os.path.join(audio_dir, f"{sound_prefix}_up.wav")

if not os.path.exists(near_sound_path):
    raise FileNotFoundError(f"Missing near sound: {near_sound_path}")
if not os.path.exists(far_sound_path):
    raise FileNotFoundError(f"Missing far sound: {far_sound_path}")

near_audio, near_sr = sf.read(near_sound_path, dtype="float32")
far_audio, far_sr = sf.read(far_sound_path, dtype="float32")
if near_sr != far_sr:
    raise ValueError("Near and far sounds must have the same sample rate.")

win = visual.Window(size=(1024, 768), fullscr=True, color="black", units="pix")
win.mouseVisible = False
kb = keyboard.Keyboard()

participant_id = ""
responses = []


def show_text(text, height=30, pos=(0, 0), color="white"):
    stim = visual.TextStim(win, text=text, color=color, height=height, wrapWidth=1200, pos=pos)
    stim.draw()
    win.flip()
    return stim


def wait_for_space_or_enter():
    event.clearEvents()
    kb.clearEvents()
    while True:
        keys = [str(k.name).lower() for k in kb.getKeys(waitRelease=False, clear=True)]
        keys.extend([str(k).lower() for k in event.getKeys()])
        if "escape" in keys:
            win.close()
            core.quit()
        if any(k in {"space", "return", "enter"} for k in keys):
            return
        core.wait(0.01)


def wait_for_space():
    event.clearEvents()
    kb.clearEvents()
    while True:
        keys = [str(k.name).lower() for k in kb.getKeys(waitRelease=False, clear=True)]
        keys.extend([str(k).lower() for k in event.getKeys()])
        if "escape" in keys:
            win.close()
            core.quit()
        if "space" in keys:
            return
        core.wait(0.01)


def normalize_key(key):
    key = str(key).lower()
    if key in {"return", "enter", "space", "backspace", "escape", "tab"}:
        return key
    if len(key) == 1:
        return key
    if key.startswith("num_") and len(key) == 5:
        return key[-1]
    if key.startswith("num") and len(key) == 4:
        return key[-1]
    if key.startswith("kp_") and len(key) == 4:
        return key[-1]
    if key.startswith("numpad") and len(key) == 7:
        return key[-1]
    return key


def get_text_input(prompt_text, audio=None, samplerate=None):
    input_str = ""
    replay_used = False
    prompt = visual.TextStim(win, text=prompt_text, color="white", height=30, wrapWidth=1200)
    while True:
        prompt.setText(f"{prompt_text}\n\n{input_str}")
        prompt.draw()
        win.flip()
        keys = [normalize_key(k) for k in event.getKeys()]
        for key in keys:
            if key == "return":
                if input_str.strip():
                    return input_str.strip(), replay_used
            elif key == "backspace":
                input_str = input_str[:-1]
            elif key == "escape":
                win.close()
                core.quit()
            elif key == "r" and audio is not None and samplerate is not None and not replay_used:
                replay_used = True
                play_sound(audio, samplerate)
            elif len(key) == 1:
                input_str += key


def get_choice_response(prompt_text, choices, audio=None, samplerate=None):
    prompt = visual.TextStim(
        win,
        text=prompt_text,
        color="white",
        height=32,
        wrapWidth=1200,
        pos=(0, 0),
    )
    replay_used = False
    while True:
        prompt.draw()
        win.flip()
        keys = []
        keys.extend([normalize_key(k.name) for k in kb.getKeys(waitRelease=False, clear=True)])
        keys.extend([normalize_key(k) for k in event.getKeys()])
        for key in keys:
            if key == "escape":
                win.close()
                core.quit()
            if key == "r" and audio is not None and samplerate is not None and not replay_used:
                replay_used = True
                play_sound(audio, samplerate)
            elif key in choices:
                return key, replay_used
        core.wait(0.01)


def play_sound(audio, samplerate):
    sd.stop()
    sd.play(audio, samplerate=samplerate, blocking=True)


def add_response(block_name, stimulus_name, response_type, response, replay_used=False):
    responses.append(
        {
            "participant_id": participant_id,
            "block": block_name,
            "stimulus": stimulus_name,
            "response_type": response_type,
            "response": response,
            "replay_used": int(bool(replay_used)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def run_free_estimation(stimulus_name, audio, samplerate, first_in_block=False):
    if first_in_block:
        show_text(
            "You will hear a sound. Please estimate how far away the sound is.\n\nPress SPACE to continue.",
            height=28,
        )
        wait_for_space_or_enter()
    else:
        show_text("Press SPACE when ready for the next sound.", height=28)
        wait_for_space()
    play_sound(audio, samplerate)
    response, replay_used = get_text_input(
        "Press R to hear the stimulus again if you would like (you can only do this once per trial) \n\nEnter a distance in meters and press ENTER:",
        audio=audio,
        samplerate=samplerate,
    )
    add_response("free_estimation", stimulus_name, "free_text", response, replay_used=replay_used)


def run_visual_estimation(stimulus_name, audio, samplerate, first_in_block=False):
    if first_in_block:
        show_text(
            "You will hear a sound. Use the 1-5 scale to indicate how near or far the sound seems.\n\n1 = nearest / 5 = farthest\n\nPress SPACE to continue.",
            height=28,
        )
        wait_for_space_or_enter()
    else:
        show_text("Press SPACE when ready for the next sound.", height=28)
        wait_for_space()
    play_sound(audio, samplerate)
    response, replay_used = get_choice_response(
        "Response options:\n\n1 = nearest\n2\n3\n4\n5 = farthest\n\n Press R to hear the stimulus again if you would like (you can only do this once per trial).\n\nPress 1-5 to answer.",
        {"1", "2", "3", "4", "5"},
        audio=audio,
        samplerate=samplerate,
    )
    add_response("visual_estimation", stimulus_name, "scale_1_5", response, replay_used=replay_used)


def run_externalisation_likert(stimulus_name, audio, samplerate, first_in_block=False):
    if first_in_block:
        show_text(
            "Externalisation Likert\n\nSometimes people experience sounds played via headphones or loudspeakers as originating inside their head or very close to them.\n\nPlease rate the sound from 1 to 5.\n\n1 = definitely within my head\n2 = very close to or on my head\n3 = it's hard to say\n4 = somewhat outside my head\n5 = definitely outside my head\n\nPress SPACE to continue.",
            height=24,
        )
        wait_for_space_or_enter()
    else:
        show_text("Press SPACE when ready for the next sound.", height=28)
        wait_for_space()
    play_sound(audio, samplerate)
    response, replay_used = get_choice_response(
        "The sound is coming from:\n\n1 = definitely within my head\n2 = very close to or on my head\n3 = it's hard to say\n4 = somewhat outside my head\n5 = definitely outside my head\n\nWhile you are responding, press R once to hear the stimulus again if you want.\n\nPress 1-5 to answer.",
        {"1", "2", "3", "4", "5"},
        audio=audio,
        samplerate=samplerate,
    )
    add_response("externalisation_likert", stimulus_name, "likert_1_5", response, replay_used=replay_used)


participant_id = get_text_input("Please enter your participant number and press ENTER:")

stimuli = [
    ("near", near_audio, near_sr),
    ("far", far_audio, far_sr),
]

for idx, (stimulus_name, audio, samplerate) in enumerate(stimuli):
    run_free_estimation(stimulus_name, audio, samplerate, first_in_block=(idx == 0))

for idx, (stimulus_name, audio, samplerate) in enumerate(stimuli):
    run_visual_estimation(stimulus_name, audio, samplerate, first_in_block=(idx == 0))

for idx, (stimulus_name, audio, samplerate) in enumerate(stimuli):
    run_externalisation_likert(stimulus_name, audio, samplerate, first_in_block=(idx == 0))

results = pd.DataFrame(responses)
results_file = os.path.join(results_dir, f"distanceestimates_{participant_id}.csv")
results.to_csv(results_file, index=False)

show_text(
    f"Finished.\n\nResponses saved to:\n{results_file}\n\nPress SPACE to exit.",
    height=28,
)
wait_for_space_or_enter()

win.close()
core.quit()