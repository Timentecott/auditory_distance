# Auditory Distance Perception Experiment

A PsychoPy-based experiment investigating participants' ability to distinguish between headphone and loudspeaker playback of spatially-rendered audio stimuli.

## Overview

This experiment presents audio stimuli through either headphones or loudspeakers and asks participants to identify the playback method. Stimuli include environmental sounds, ISTS (International Speech Test Signal), and noise, with binaural/spatial processing applied for headphone presentation.

## Requirements

- Python 3.11 (via conda)
- PsychoPy 3.2.4+
- Audio playback hardware:
  - Headphones (device index configurable)
  - Loudspeakers (device index configurable)

See `requirements.txt` for full dependencies.

## Setup

### 1. Create and activate conda environment

```bash
conda create -n pilot python=3.11
conda activate pilot
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure audio devices

Edit `pilot_exp.py` lines 213-214 to match your audio device indices:

```python
headphones_device = 13  # Change to your headphone device index
speakers_device = 11    # Change to your speaker device index
```

To find your device indices, run:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## Running the Experiment

1. Activate the conda environment:
   ```bash
   conda activate pilot
   ```

2. Run the experiment script:
   ```bash
   python pilot_exp.py
   ```

3. Follow on-screen instructions:
   - Enter participant ID
   - Complete 3 practice trials
   - Complete 3 blocks of experimental trials (12 total)
   - Use **UP ARROW** for loudspeaker, **DOWN ARROW** for headphone

## Project Structure

```
├── pilot_exp.py              # Main experiment script
├── requirements.txt          # Python dependencies
├── localised_stimuli/        # Binaural/HRTF-processed audio for headphones
│   ├── environment/
│   ├── ISTS/
│   └── noise/
├── loudspeaker_stimuli/      # Mono/stereo audio for speaker playback
│   ├── environment/
│   ├── ISTS/
│   └── noise/
├── resources/                # Images and other resources
│   └── headphonevsloudspeak_info_graphic.png
└── results/                  # Output CSV files (one per participant)
```

## Experiment Design

- **Practice trials**: 3 (with visual aid)
- **Experimental trials**: 12 (4 per block, 3 blocks)
- **Stimulus duration**: 5 seconds (truncated)
- **Random gain**: ±10 dB per trial
- **Safety limiter**: Hard limit at -9 dBFS (≈85 dB SPL)
- **Response method**: Arrow keys (up = speaker, down = headphone)

### Trial Balance

- 50% headphone, 50% loudspeaker
- Equal distribution across stimulus categories (environment, ISTS, noise)
- Randomized trial order

## Output

Results are saved to `results/<participant_id>.csv` with columns:
- `presentation_type`: 'headphone' or 'speaker' (with '(practice)' suffix for practice trials)
- `stimulus`: filename of played audio
- `stimulus_category`: 'environment', 'ISTS', or 'noise'
- `response`: 'up' or 'down'
- `rt`: response time (seconds)
- `accuracy`: 1 = correct, 0 = incorrect
- `timestamp`: trial completion time

## Safety Features

- **Hard limiter**: Prevents audio exceeding -9 dBFS (≈85 dB SPL)
- **Gain randomization**: Applied before safety check
- **Mono conversion**: Prevents channel mismatches

## Troubleshooting

### Audio not playing through correct device
- Verify device indices with `sd.query_devices()`
- Check that devices are set as default in Windows Sound settings
- Restart the experiment after changing device configuration

### PsychoPy window issues
- Fullscreen mode may conflict with multi-monitor setups
- Set `fullscr=False` in `visual.Window()` to run in windowed mode

### Permission errors
- Run terminal as administrator if device access fails
- Check that audio devices aren't locked by other applications

## Notes

- Audio files are not included in `.gitignore` by default (change if files are large)
- Results CSVs are excluded from version control
- The `env/` virtual environment folder is ignored

## License

[Add license information]

## Contact

[Add contact information]
