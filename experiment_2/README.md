# Posner Task

This folder contains the Posner cueing task and related audio utilities.

## Main files

- `posner_pilot_00.py` — main Posner task experiment
- `check_input_output_index.py` — helper script to inspect and test audio devices
- `localise_udlr.py` — generates localised UDLR audio from SOFA/BRIR data
- `individual_analysis.py` — analysis for indvidual participant results
- `posner_for_pav.py` — alternate experiment version for pavlovia (not currenlty working)

## Requirements

Install the Python packages listed in `requirements.txt`.

PsychoPy is the main runtime dependency. If `pip install psychopy` does not work on your machine, use the official PsychoPy installer or a conda environment recommended by PsychoPy.

## Setup

1. Copy the entire `posner` folder to the other computer.
2. Keep the folder structure unchanged.
3. Make sure the following folders exist relative to the scripts:
   - `audio_stimuli/`
   - `visual stimuli/`
   - `resources/`

## Running the main task

Run:

```bash
python posner_pilot_00.py
```

## Audio device checking

If you need to find the correct input/output device indices, run:

```bash
python check_input_output_index.py
```

## Localised audio generation

`localise_udlr.py` is used to generate localised audio files. It currently uses a fixed SOFA file path in the source code, so if you move the folder to another machine you may need to update those path settings.

## Output

The experiment saves trial data into CSV files from the script directory.


