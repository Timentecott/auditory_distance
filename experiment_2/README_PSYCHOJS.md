# Posner Cueing Task - PsychoJS Version

Web-compatible version of the Posner spatial cueing task for online administration via Pavlovia.

## Features

- **Practice block** (4 trials): valid/invalid cue validity with feedback
- **Main block** (64 trials): balanced across 4 locations × 2 validity conditions
- **Spatial audio cues**: directional sound files from `audio_stimuli/Localised/`
- **Visual response**: arrow keys (left/right/up/down)
- **Data logging**: automatic CSV export via Pavlovia

## Setup for Pavlovia

1. **Create a Pavlovia project**: https://pavlovia.org
2. **Link your repo**:
   ```
   git remote add pavlovia https://gitlab.pavlovia.org/<username>/posner_cueing.git
   ```

3. **Folder structure** (required):
   ```
   posner/
   ├── index.html
   ├── posner_pilot_00.js
   └── audio_stimuli/
       └── Localised/
           ├── pink_noise_48k_30s_300_8000hz_left.wav
           ├── pink_noise_48k_30s_300_8000hz_right.wav
           ├── pink_noise_48k_30s_300_8000hz_up.wav
           └── pink_noise_48k_30s_300_8000hz_down.wav
   ```

4. **Sync to Pavlovia**:
   ```
   git add -A
   git commit -m "Add Posner task PsychoJS version"
   git push pavlovia main
   ```

5. **Test online**: Visit your Pavlovia project link → click **Pilot** to run

## Files

- `index.html`: Entry point (loads PsychoJS libraries and main script)
- `posner_pilot_00.js`: Experiment logic (trials, timing, responses, data logging)

## Timing (seconds)

- Fixation: 0.5
- Sound cue: 0.1
- Cue-to-dot ISI: 0.1
- Dot display: 0.1
- Response window: unlimited (until keypress)
- Post-trial pause: 0.5
- Inter-trial interval: 1.0

## Trial Structure

**Practice (4 trials with feedback)**:
- Sound location (random): left, right, up, down
- Dot location: valid (same as sound) or invalid (opposite)
- Feedback: "Correct!" (green) or "Incorrect" (red)

**Main (64 trials, no feedback)**:
- 16 valid trials per location
- 16 invalid trials per location
- Counterbalanced and shuffled

## Notes

- **Web compatibility**: Runs in Chrome, Firefox, Safari (Edge supported)
- **Audio**: Uses Web Audio API; ensure sound files are .wav format
- **Data**: Collected per-trial with timestamps, exported as JSON/CSV via Pavlovia dashboard
- **Keyboard**: Respects platform arrow-key naming conventions
- **Performance**: ~60 FPS target; minimal latency overhead

## Troubleshooting

**Audio not playing**:
- Check file paths relative to `posner/` folder
- Verify `.wav` files exist in `audio_stimuli/Localised/`
- Browser security: HTTPS required on live sites

**Keys not registering**:
- Some browsers require focus on canvas/window
- Test in Chrome developer console

**Data not saving**:
- Ensure Pavlovia credentials are set in experiment settings
- Check Pavlovia dashboard for logs

## Local Testing

For local testing (without Pavlovia):
```bash
cd posner
python -m http.server 8000
# Open http://localhost:8000/index.html
```

Note: Some features (data persistence, full sound loading) require Pavlovia hosting.
