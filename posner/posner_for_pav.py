from psychopy import visual, event, core, sound, data
import random


# Experiment parameters
NUMBER_OF_TRIALS = 64  # divisible by 8 (4 locations x 2 validity)
FIXATION_DURATION = 0.5
SOUND_CUE_DURATION = 0.1
CUE_TO_DOT_ISI = 0.1
DOT_DURATION = 0.1
INTER_TRIAL_INTERVAL = 1.0
PRACTICE_TRIALS = 4

# Resource paths (relative to experiment root for Pavlovia)
SOUND_PREFIX = "audio_stimuli/Localised/pink_noise_48k_30s_300_8000hz"
PRACTICE_IMAGE = "visual stimuli/numpad_pic.png"

# Start keys: include numpad variants + fallbacks for browser differences
START_KEYS = {
    "num_5", "num5", "kp_5", "numpad5", "5", "clear", "space", "return"
}
RESPONSE_KEYS = ["left", "right", "up", "down", "escape"]

# Coordinates
DISTANCE_FROM_CENTER = 200
LOCATIONS = {
    "left": (-DISTANCE_FROM_CENTER, 0),
    "right": (DISTANCE_FROM_CENTER, 0),
    "up": (0, DISTANCE_FROM_CENTER),
    "down": (0, -DISTANCE_FROM_CENTER),
}


def opposite_location(location):
    if location == "left":
        return "right"
    if location == "right":
        return "left"
    if location == "up":
        return "down"
    return "up"


def wait_for_start_key(win):
    event.clearEvents()
    while True:
        keys = event.getKeys()
        if keys:
            key_names = [str(k).strip().lower() for k in keys]
            if "escape" in key_names:
                win.close()
                core.quit()
            if any((k in START_KEYS) or ("5" in k) for k in key_names):
                return
        core.wait(0.01)


def make_main_trials(n_trials):
    location_names = ["left", "right", "up", "down"]
    trials_per_location = n_trials // 4
    trials_per_validity = n_trials // 2

    sound_locations = []
    for loc in location_names:
        sound_locations.extend([loc] * trials_per_location)

    trial_validity = [True] * trials_per_validity + [False] * trials_per_validity

    random.shuffle(sound_locations)
    random.shuffle(trial_validity)

    trials = []
    for i in range(n_trials):
        sound_location = sound_locations[i]
        is_valid = trial_validity[i]
        dot_location = sound_location if is_valid else opposite_location(sound_location)
        sound_file = f"{SOUND_PREFIX}_{sound_location}.wav"
        trials.append(
            {
                "trial_number": i + 1,
                "sound_location": sound_location,
                "dot_location": dot_location,
                "is_valid": is_valid,
                "sound_file": sound_file,
            }
        )

    return trials


def make_practice_trials():
    practice_locations = ["left", "right", "up", "down"]
    trials = []
    for i, loc in enumerate(practice_locations):
        is_valid = (i % 2 == 0)
        dot_location = loc if is_valid else opposite_location(loc)
        sound_file = f"{SOUND_PREFIX}_{loc}.wav"
        trials.append(
            {
                "trial_number": i + 1,
                "sound_location": loc,
                "dot_location": dot_location,
                "is_valid": is_valid,
                "sound_file": sound_file,
            }
        )

    random.shuffle(trials)
    return trials[:PRACTICE_TRIALS]


def run_trial(win, fixation, dot, trial, show_feedback=False):
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)

    trial_sound = sound.Sound(trial["sound_file"])
    trial_sound.play()
    core.wait(SOUND_CUE_DURATION)
    trial_sound.stop()

    win.flip()
    core.wait(CUE_TO_DOT_ISI)

    dot.pos = LOCATIONS[trial["dot_location"]]
    dot.draw()
    win.flip()
    core.wait(DOT_DURATION)

    win.flip()

    response_start = core.getTime()
    response = None
    response_time = None

    while response is None:
        keys = event.getKeys(keyList=RESPONSE_KEYS)
        if keys:
            response = keys[0]
            response_time = core.getTime() - response_start

    if response == "escape":
        win.close()
        core.quit()

    correct = response == trial["dot_location"]

    if show_feedback:
        feedback = visual.TextStim(
            win,
            text="Correct!" if correct else "Incorrect",
            color="green" if correct else "red",
            height=40,
        )
        feedback.draw()
        win.flip()
        core.wait(0.5)
    else:
        core.wait(0.5)

    win.flip()
    core.wait(INTER_TRIAL_INTERVAL)

    return response, response_time, correct


def main():
    exp = data.ExperimentHandler(name="posner_for_pav")

    win = visual.Window(size=[1920, 1080], fullscr=True, color="gray", units="pix")
    win.mouseVisible = False

    fixation = visual.ShapeStim(
        win,
        vertices="cross",
        size=30,
        lineColor="white",
        fillColor="white",
    )

    dot = visual.Circle(
        win,
        radius=20,
        fillColor="white",
        lineColor="white",
    )

    practice_instructions = visual.TextStim(
        win,
        text=(
            "You will hear a sound, then see a dot.\n"
            "Press an arrow key (LEFT, RIGHT, UP, DOWN) to indicate where the dot is.\n"
            "Use one finger for the whole experiment.\n\n"
            "Press 5 on the numpad to start practice (space/return also accepted)."
        ),
        color="white",
        height=30,
        wrapWidth=1000,
        pos=(0, 180),
    )

    practice_image = visual.ImageStim(
        win,
        image=PRACTICE_IMAGE,
        pos=(0, -120),
        size=(500, 350),
    )

    main_instructions = visual.TextStim(
        win,
        text=(
            "Practice complete.\n"
            "Press 5 to start the main experiment.\n"
            "No feedback will be shown in the main block."
        ),
        color="white",
        height=30,
        wrapWidth=1000,
    )

    end_text = visual.TextStim(
        win,
        text="Experiment complete!\n\nThank you for participating.",
        color="white",
        height=30,
    )

    practice_instructions.draw()
    practice_image.draw()
    win.flip()
    wait_for_start_key(win)

    for trial in make_practice_trials():
        run_trial(win, fixation, dot, trial, show_feedback=True)

    main_instructions.draw()
    win.flip()
    wait_for_start_key(win)

    main_trials = make_main_trials(NUMBER_OF_TRIALS)
    for trial in main_trials:
        response, rt, correct = run_trial(win, fixation, dot, trial, show_feedback=False)

        exp.addData("trial_number", trial["trial_number"])
        exp.addData("sound_location", trial["sound_location"])
        exp.addData("dot_location", trial["dot_location"])
        exp.addData("is_valid", trial["is_valid"])
        exp.addData("sound_file", trial["sound_file"])
        exp.addData("response", response)
        exp.addData("correct", correct)
        exp.addData("response_time", rt)
        exp.nextEntry()

    end_text.draw()
    win.flip()
    core.wait(2)

    win.close()
    core.quit()


if __name__ == "__main__":
    main()
