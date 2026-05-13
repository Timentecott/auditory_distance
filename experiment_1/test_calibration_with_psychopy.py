#!/usr/bin/env python
"""Test calibration with a minimal PsychoPy window."""

import os
os.environ["SD_ENABLE_ASIO"] = "1"

from psychopy import visual, event, core
import sounddevice as sd
import numpy as np
import sys

# Minimal setup
ASIO_AGGREGATE_DEVICE = 16
sample_rate = 48000

def ensure_stereo(audio):
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.column_stack([audio[:, 0], audio[:, 0]])
    return audio[:, :2]

def play_audio_on_device(audio, sample_rate, device_index, mapping=None):
    audio = np.asarray(audio, dtype=np.float32)
    print(f"  [play_audio_on_device] device={device_index}, mapping={mapping}, shape={audio.shape}")
    sd.play(audio, samplerate=sample_rate, device=device_index, mapping=mapping)
    sd.wait()
    print(f"  [play_audio_on_device] playback complete")

# Create minimal PsychoPy window
print("Creating PsychoPy window...")
try:
    win = visual.Window(
        size=(1024, 768),
        units='pix',
        fullscr=False,
        color=(0, 0, 0),
        allowStencil=False
    )
    print("✓ Window created")
except Exception as e:
    print(f"✗ Window creation failed: {e}")
    sys.exit(1)

# Show instruction
instruction = visual.TextStim(
    win,
    text="Calibration test with PsychoPy window.\n\nPress any key to begin audio playback.",
    color='white',
    height=28
)
instruction.draw()
win.flip()
event.waitKeys()

# Create test audio
rng = np.random.default_rng(42)
test_audio = rng.standard_normal(int(sample_rate * 1.0)).astype(np.float32)
test_audio = ensure_stereo(test_audio).astype(np.float32)

# Set latency
sd.default.latency = 'low'

# Test speaker playback
print("\nTesting speaker playback (channels 1-2)...")
try:
    instruction.setText("Playing speaker audio (channels 1-2)...\n\nListening...")
    instruction.draw()
    win.flip()
    play_audio_on_device(test_audio, sample_rate, ASIO_AGGREGATE_DEVICE, mapping=[1, 2])
    print("✓ Speaker playback successful")
except Exception as e:
    print(f"✗ Speaker playback failed: {e}")
    instruction.setText(f"ERROR: Speaker playback failed!\n{e}")
    instruction.draw()
    win.flip()
    event.waitKeys()
    win.close()
    core.quit()
    sys.exit(1)

# Test headphone playback
print("\nTesting headphone playback (channels 3-4)...")
try:
    instruction.setText("Playing headphone audio (channels 3-4)...\n\nListening...")
    instruction.draw()
    win.flip()
    play_audio_on_device(test_audio, sample_rate, ASIO_AGGREGATE_DEVICE, mapping=[3, 4])
    print("✓ Headphone playback successful")
except Exception as e:
    print(f"✗ Headphone playback failed: {e}")
    instruction.setText(f"ERROR: Headphone playback failed!\n{e}")
    instruction.draw()
    win.flip()
    event.waitKeys()
    win.close()
    core.quit()
    sys.exit(1)

# Success
instruction.setText("All tests passed!\n\nPress any key to close.")
instruction.draw()
win.flip()
event.waitKeys()

win.close()
core.quit()

print("\n" + "="*70)
print("✓ Calibration with PsychoPy window test passed!")
print("="*70)
