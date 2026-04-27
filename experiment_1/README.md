# Experiment 1

This folder contains the PsychoPy headphone/loudspeaker experiment and the audio resources it needs.

## Files

- `pilot_exp.py` — main experiment script
- `check_input_output_index.py` — helper script to check audio devices
- `level_detector.py` — microphone level utility, not working well
- `eq_to_ear.py` — audio helper script, not working well
- `resources/` — images and SOFA/BRIR files used by the experiment
- `localised_stimuli_b20/` — headphone stimuli
- `loudspeaker_stimuli/` — speaker stimuli
- `results/` — created automatically when the experiment runs

## Requirements

I use a conda venv with Python 3.10

Install the Python packages in `requirements.txt`.

PsychoPy is the main dependency. If `pip install psychopy` does not work on your machine, install PsychoPy using the official PsychoPy installer or conda environment recommended by PsychoPy.


## Running the experiment

Run:

```bash
python pilot_exp.py
```

## Audio device selection

output devices must be selected in the script. 

If you do not know the correct indices, run:

```bash
python check_input_output_index.py
```

and use the listed output device numbers.


## Output files

Results are saved automatically into the `results/` folder as CSV files:

- `*_demographics.csv`
- `*_trials.csv`

## Notes

- The experiment currently uses keyboard responses only.
- If you change the stimulus folder names, update the script accordingly.
- If you add or remove stimuli, keep the folder naming convention the same.
