#!/usr/bin/env python
"""Minimal test to verify ASIO playback works for the calibration."""

import os
os.environ["SD_ENABLE_ASIO"] = "1"
import sounddevice as sd
import numpy as np

# Test device 16 with channels 1/2 and 3/4
DEVICE = 16
SAMPLE_RATE = 48000
DURATION = 1.0

# Generate white noise reference
rng = np.random.default_rng(42)
reference_audio = rng.standard_normal(int(SAMPLE_RATE * DURATION)).astype(np.float32)

# Scale to target RMS
target_rms = 0.01
cur_rms = np.sqrt(np.mean(np.square(reference_audio.astype(np.float64))))
if cur_rms > 0:
    reference_audio = (reference_audio.astype(np.float64) * (target_rms / cur_rms)).astype(np.float32)

# Ensure stereo
if reference_audio.ndim == 1:
    reference_audio = np.column_stack([reference_audio, reference_audio])

# Set low latency
sd.default.latency = 'low'

# Test speaker output (channels 1-2)
print("\n" + "="*60)
print("Testing ASIO device 16 channel mapping...")
print("="*60)

print("\nTesting loudspeaker (channels 1-2)...")
try:
    sd.play(reference_audio, samplerate=SAMPLE_RATE, device=DEVICE, mapping=[1, 2])
    sd.wait()
    print("[OK] Loudspeaker playback successful!")
except Exception as e:
    print(f"[FAIL] Loudspeaker playback failed: {e}")
    exit(1)

print("\nTesting headphone (channels 3-4)...")
try:
    sd.play(reference_audio, samplerate=SAMPLE_RATE, device=DEVICE, mapping=[3, 4])
    sd.wait()
    print("[OK] Headphone playback successful!")
except Exception as e:
    print(f"[FAIL] Headphone playback failed: {e}")
    exit(1)

print("\n" + "="*60)
print("All ASIO tests passed!")
print("="*60)
