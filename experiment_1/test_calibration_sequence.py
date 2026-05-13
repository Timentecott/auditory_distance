#!/usr/bin/env python
"""Minimal test of the experiment's calibration sequence without PsychoPy window."""

import os
os.environ["SD_ENABLE_ASIO"] = "1"

# Now import sounddevice and other libraries
import sounddevice as sd
import numpy as np
import sys

# Minimal setup mirroring the experiment
ASIO_AGGREGATE_DEVICE = 16
ASIO_SPEAKER_MAPPING = [1, 2]
ASIO_HEADPHONE_MAPPING = [3, 4]
headphones_device = ASIO_AGGREGATE_DEVICE
speakers_device = ASIO_AGGREGATE_DEVICE
sample_rate = 48000

def ensure_stereo(audio):
    """Force audio to stereo (L/R) for headphone playback."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.column_stack([audio[:, 0], audio[:, 0]])
    return audio[:, :2]

def apply_gain_db(audio, gain_db):
    """Apply gain in dB to an audio array."""
    return audio * (10 ** (gain_db / 20.0))

def collapse_to_left_channel(audio, preserve_total_rms=True):
    """Collapse audio to left channel only for loudspeaker playback."""
    if audio is None:
        return audio

    try:
        audio_f = audio.astype(np.float64)
    except Exception:
        audio_f = np.array(audio, dtype=np.float64)

    if audio_f.size == 0:
        if audio.ndim == 1:
            return np.zeros((0, 2), dtype=audio.dtype)
        return np.zeros((0, 2), dtype=audio.dtype)

    orig_rms = np.sqrt(np.mean(np.square(audio_f)))

    if audio.ndim == 1:
        stereo = np.zeros((len(audio), 2), dtype=audio.dtype)
        stereo[:, 0] = audio
    elif audio.shape[1] >= 2:
        stereo = np.zeros_like(audio[:, :2])
        stereo[:, 0] = audio[:, 0]
    else:
        stereo = np.zeros((audio.shape[0], 2), dtype=audio.dtype)
        stereo[:, 0] = audio[:, 0]

    if preserve_total_rms:
        stereo_f = stereo.astype(np.float64)
        new_rms = np.sqrt(np.mean(np.square(stereo_f)))
        if new_rms > 0 and orig_rms > 0:
            scale = float(orig_rms / new_rms)
            stereo = (stereo_f * scale).astype(audio.dtype)

    return stereo

def apply_fade(audio, sample_rate, fade_ms=10):
    """Apply short fade-in/out to reduce clicks at playback boundaries."""
    fade_samples = int(sample_rate * fade_ms / 1000.0)
    if fade_samples <= 0:
        return audio

    n_samples = audio.shape[0]
    if n_samples < 2 * fade_samples:
        fade_samples = n_samples // 2
    if fade_samples <= 0:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=True)
    fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=True)

    audio = audio.copy()
    if audio.ndim == 1:
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    else:
        audio[:fade_samples, :] *= fade_in[:, None]
        audio[-fade_samples:, :] *= fade_out[:, None]
    return audio

def play_audio_on_device(audio, sample_rate, device_index, mapping=None):
    """Play a block of audio through a device."""
    audio = np.asarray(audio, dtype=np.float32)
    print(f"  [play_audio_on_device] device={device_index}, mapping={mapping}, shape={audio.shape}, dtype={audio.dtype}")
    sd.play(audio, samplerate=sample_rate, device=device_index, mapping=mapping)
    sd.wait()
    print(f"  [play_audio_on_device] playback complete")

# Main test
print("\n" + "="*70)
print("Testing experiment calibration sequence (no PsychoPy window)")
print("="*70)

# Create reference audio (same as in calibration)
rng = np.random.default_rng(42)
duration_s = 1.0
reference_audio = rng.standard_normal(int(sample_rate * duration_s)).astype(np.float32)
target_rms = 0.01
cur_rms = np.sqrt(np.mean(np.square(reference_audio.astype(np.float64))))
if cur_rms > 0:
    reference_audio = (reference_audio.astype(np.float64) * (target_rms / cur_rms)).astype(np.float32)

reference_headphone_eq = ensure_stereo(reference_audio).astype(np.float32)

# Set latency
sd.default.latency = 'low'

# Create speaker audio
speaker_audio = apply_gain_db(reference_audio, 0.0)
speaker_audio = collapse_to_left_channel(speaker_audio, preserve_total_rms=True)
speaker_audio = apply_fade(speaker_audio, sample_rate, fade_ms=20)
speaker_audio = ensure_stereo(speaker_audio).astype(np.float32)

# Test speaker playback
print("\nTest 1: Speaker (channels 1-2)...")
try:
    play_audio_on_device(speaker_audio, sample_rate, speakers_device, mapping=[1, 2])
    print("[OK] Speaker test passed")
except Exception as e:
    print(f"[FAIL] Speaker test failed: {e}")
    sys.exit(1)

# Create headphone audio with offset
headphone_offset_db = 3.0
headphone_audio = apply_gain_db(reference_headphone_eq, headphone_offset_db)
headphone_audio = apply_fade(headphone_audio, sample_rate, fade_ms=20)
headphone_audio = ensure_stereo(headphone_audio).astype(np.float32)

# Test headphone playback
print("\nTest 2: Headphone (channels 3-4)...")
try:
    play_audio_on_device(headphone_audio, sample_rate, headphones_device, mapping=[3, 4])
    print("[OK] Headphone test passed")
except Exception as e:
    print(f"[FAIL] Headphone test failed: {e}")
    sys.exit(1)

# Test alternating playback (like calibration loop)
print("\nTest 3: Alternating sequence (3 cycles)...")
try:
    for cycle in range(3):
        print(f"\n  Cycle {cycle + 1}/3:")
        print(f"    Playing speaker...")
        play_audio_on_device(speaker_audio, sample_rate, speakers_device, mapping=[1, 2])
        print(f"    Playing headphone...")
        play_audio_on_device(headphone_audio, sample_rate, headphones_device, mapping=[3, 4])
    print("[OK] Alternating sequence test passed")
except Exception as e:
    print(f"[FAIL] Alternating sequence test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("All calibration sequence tests passed!")
print("="*70)
