#!/usr/bin/env python3
"""Create localized stimuli from a single dry source audio and four RIRs.

This script takes one dry audio stimulus and four RIRs (in-situ-near, in-situ-far, ex-situ-near, ex-situ-far)
and creates 5 output files:
- A loudspeaker version (mono audio in left channel only)
- Four spatially localized versions (one for each RIR)

Processing includes:
- Bandpass filtering (100 Hz - 18 kHz)
- Linear fade-in/fade-out ramps
- Silence padding
- RMS normalization to -20 dBFS
- Peak limiting
- Optional IR tail fade for reduced echoiness

Example:
    python master_stim_create_2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy import signal

# ============================================================================
# CONFIGURATION: Edit these variables to change input/output paths and RIRs
# ============================================================================

# Path to the dry audio stimulus file (.wav)
STIMULUS_PATH = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\short_tap.wav"

# Output directory where convolved audio files will be saved
OUTPUT_DIR = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\Localised"

# Dictionary of RIR file paths (.npy)
# Keys must be: 'in-situ-near', 'in-situ-far', 'ex-situ-near', 'ex-situ-far'
RIR_PATHS = {
    'in-situ-near': r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\resources\insitu_near_2409\RIR.npy",
    'in-situ-far': r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\resources\insitu_far_2409\RIR.npy",
    'ex-situ-near': r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\resources\exsitu_near_2309\RIR.npy",
    'ex-situ-far': r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\resources\exsitu_far_2309\RIR.npy",
}

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

DEFAULT_RIR_SR = 44100
OUTPUT_SAMPLE_RATE = 44100

LOUDSPEAKER_TARGET_RMS_DBFS = -20.0
SPATIAL_TARGET_RMS_DBFS = -20.0
LOUDSPEAKER_TARGET_RMS = 10.0 ** (LOUDSPEAKER_TARGET_RMS_DBFS / 20.0)
SPATIAL_TARGET_RMS = 10.0 ** (SPATIAL_TARGET_RMS_DBFS / 20.0)
MAX_PEAK = 0.999

# Filtering parameters
BANDPASS_LOW_HZ = 100.0
BANDPASS_HIGH_HZ = 18000.0
BANDPASS_ORDER = 4

# Audio shaping parameters
SILENCE_PAD_SECONDS = 0.5
CLICK_RAMP_SECONDS = 0.05

# Processing settings
PRESERVE_RMS = False  # Change to True if needed
IR_TAIL_FADE_MS = None  # Change to a value like 150.0 if needed
RIR_SAMPLE_RATE = DEFAULT_RIR_SR


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono by averaging channels if stereo, or return as-is if already mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def mono_left_only_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono audio to stereo with audio routed to left channel only."""
    mono = ensure_mono(audio)
    stereo = np.zeros((mono.shape[0], 2), dtype=np.float32)
    stereo[:, 0] = mono.astype(np.float32)
    return stereo


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono or multichannel audio to target_sr."""
    if orig_sr == target_sr:
        return audio

    ratio = float(target_sr) / float(orig_sr)
    n_samples = int(round(audio.shape[0] * ratio))

    if audio.ndim == 1:
        return signal.resample(audio, n_samples)

    resampled_channels = []
    for ch in range(audio.shape[1]):
        resampled_channels.append(signal.resample(audio[:, ch], n_samples))
    return np.column_stack(resampled_channels)


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio from file."""
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def normalize_audio(audio: np.ndarray, target_rms: float, max_peak: float = MAX_PEAK) -> np.ndarray:
    """Normalize audio to target RMS and prevent clipping."""
    audio = np.asarray(audio, dtype=np.float32)
    current_rms = compute_rms(audio)
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > max_peak:
        audio = audio * (max_peak / peak)
    return audio.astype(np.float32)


def apply_ir_tail_fade(
    rir: np.ndarray, rir_sr: int, fade_duration_ms: float
) -> np.ndarray:
    """Apply exponential fade-out to the tail of an IR to reduce echoiness.

    Args:
        rir: IR array with shape (n_samples, n_channels)
        rir_sr: Sample rate of the IR
        fade_duration_ms: Duration of fade-out in milliseconds

    Returns:
        Faded IR array
    """
    rir = np.asarray(rir, dtype=np.float32)
    fade_samples = int(rir_sr * fade_duration_ms / 1000.0)

    if fade_samples >= rir.shape[0]:
        return rir

    fade_start = rir.shape[0] - fade_samples
    fade_curve = np.exp(np.linspace(0, -5, fade_samples))  # exponential decay from 1 to ~0.007

    rir_faded = rir.copy()
    if rir.ndim == 1:
        rir_faded[fade_start:] *= fade_curve
    else:
        for ch in range(rir.shape[1]):
            rir_faded[fade_start:, ch] *= fade_curve

    return rir_faded


def load_rir_array(rir_path: Path, rir_sr: int) -> tuple[np.ndarray, int]:
    """Load an RIR from .npy file and reshape to (n_samples, n_channels).

    Args:
        rir_path: Path to RIR file (.npy)
        rir_sr: Sample rate for the RIR

    Returns:
        tuple: (rir_array, sample_rate) where rir_array has shape (n_samples, n_channels)
    """
    rir_path = Path(rir_path)
    if rir_path.suffix.lower() != ".npy":
        raise ValueError(f"Unsupported RIR format: {rir_path.suffix}. Use .npy")

    rir = np.load(str(rir_path))
    rir = np.asarray(rir, dtype=np.float32)

    # Reshape 1D to (n_samples, 1) if needed
    if rir.ndim == 1:
        rir = rir[:, None]
    elif rir.ndim == 2:
        pass  # Already (n_samples, n_channels)
    else:
        raise ValueError(f"RIR array must be 1D or 2D, got shape {rir.shape}")

    return rir, int(rir_sr)


def convolve_with_rir(source_mono: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve a mono source with an RIR array.

    Args:
        source_mono: Mono audio source, shape (n_samples,)
        rir: Room impulse response, shape (n_samples, n_channels)

    Returns:
        Convolved audio with shape (n_out_samples, n_channels)
    """
    if source_mono.ndim != 1:
        raise ValueError("Source audio for convolution must be mono")

    if rir.ndim == 1:
        rir = rir[:, None]

    n_channels = rir.shape[1]
    out_len = source_mono.shape[0] + rir.shape[0] - 1
    output = np.zeros((out_len, n_channels), dtype=np.float32)
    for ch in range(n_channels):
        output[:, ch] = signal.fftconvolve(source_mono, rir[:, ch], mode="full")
    return output


def apply_bandpass_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply bandpass filter to audio."""
    audio = np.asarray(audio, dtype=np.float32)
    nyquist = sample_rate / 2.0
    if BANDPASS_LOW_HZ <= 0 or BANDPASS_HIGH_HZ >= nyquist:
        raise ValueError(
            f"Bandpass cutoff frequencies must satisfy 0 < {BANDPASS_LOW_HZ} < {BANDPASS_HIGH_HZ} < Nyquist ({nyquist} Hz)"
        )

    sos = signal.butter(
        BANDPASS_ORDER,
        [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    if audio.ndim == 1:
        try:
            return signal.sosfiltfilt(sos, audio).astype(np.float32)
        except ValueError:
            return signal.sosfilt(sos, audio).astype(np.float32)

    filtered_channels = []
    for ch in range(audio.shape[1]):
        channel = audio[:, ch]
        try:
            filtered = signal.sosfiltfilt(sos, channel)
        except ValueError:
            filtered = signal.sosfilt(sos, channel)
        filtered_channels.append(filtered.astype(np.float32))
    return np.column_stack(filtered_channels)


def apply_linear_ramps(audio: np.ndarray, sample_rate: int, ramp_seconds: float) -> np.ndarray:
    """Apply linear fade-in and fade-out ramps to audio."""
    audio = np.asarray(audio, dtype=np.float32)
    ramp_samples = int(round(sample_rate * ramp_seconds))
    if ramp_samples <= 0 or audio.shape[0] == 0:
        return audio

    ramp_samples = min(ramp_samples, audio.shape[0] // 2)
    if ramp_samples <= 0:
        return audio

    ramp = np.linspace(0.0, 1.0, ramp_samples, endpoint=True, dtype=np.float32)
    shaped = audio.copy()

    if shaped.ndim == 1:
        shaped[:ramp_samples] *= ramp
        shaped[-ramp_samples:] *= ramp[::-1]
    else:
        shaped[:ramp_samples, :] *= ramp[:, None]
        shaped[-ramp_samples:, :] *= ramp[::-1][:, None]
    return shaped


def add_silence_padding(audio: np.ndarray, sample_rate: int, pad_seconds: float) -> np.ndarray:
    """Add silence padding before and after audio."""
    audio = np.asarray(audio, dtype=np.float32)
    pad_samples = int(round(sample_rate * pad_seconds))
    if pad_samples <= 0:
        return audio

    if audio.ndim == 1:
        padding = np.zeros(pad_samples, dtype=np.float32)
        return np.concatenate([padding, audio, padding])

    padding = np.zeros((pad_samples, audio.shape[1]), dtype=np.float32)
    return np.vstack([padding, audio, padding])


def apply_final_stimulus_processing(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply final processing: bandpass filter, ramps, and silence padding."""
    filtered = apply_bandpass_filter(audio, sample_rate)
    ramped = apply_linear_ramps(filtered, sample_rate, CLICK_RAMP_SECONDS)
    return add_silence_padding(ramped, sample_rate, SILENCE_PAD_SECONDS)


def process_loudspeaker(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Process audio for loudspeaker playback (mono)."""
    mono = ensure_mono(audio)
    mono = normalize_audio(mono, LOUDSPEAKER_TARGET_RMS)
    return resample_audio(mono, sample_rate, OUTPUT_SAMPLE_RATE).astype(np.float32)


def process_spatial(
    audio: np.ndarray,
    sample_rate: int,
    rir: np.ndarray,
    rir_sr: int,
    preserve_rms: bool = False,
    max_amp: float = MAX_PEAK,
    ir_tail_fade_duration_ms: float | None = None,
) -> np.ndarray:
    """Process audio with spatial RIR convolution."""
    mono = ensure_mono(audio)

    if sample_rate != rir_sr:
        mono = resample_audio(mono, sample_rate, rir_sr)
        sample_rate = rir_sr

    orig_rms = compute_rms(mono)

    rir_to_use = rir
    if ir_tail_fade_duration_ms is not None and ir_tail_fade_duration_ms > 0:
        rir_to_use = apply_ir_tail_fade(rir, rir_sr, ir_tail_fade_duration_ms)

    localized = convolve_with_rir(mono, rir_to_use)

    if preserve_rms and orig_rms > 0:
        new_rms = compute_rms(localized)
        if new_rms > 0:
            localized = localized * (orig_rms / new_rms)

    localized = normalize_audio(localized, SPATIAL_TARGET_RMS, max_peak=max_amp)
    return resample_audio(localized, rir_sr, OUTPUT_SAMPLE_RATE).astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Apply final processing and save audio to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = apply_final_stimulus_processing(audio, sample_rate)
    sf.write(str(path), audio, sample_rate)


def create_localized_stimuli(
    stimulus_path: str,
    rir_paths_dict: dict,
    output_dir: str = 'recordings'
) -> None:
    """Create localized stimuli from a single dry stimulus and four RIRs.

    Args:
        stimulus_path: Path to dry audio stimulus file (.wav)
        rir_paths_dict: Dictionary with RIR labels as keys and paths as values
                       Expected keys: 'in-situ-near', 'in-situ-far', 'ex-situ-near', 'ex-situ-far'
        output_dir: Directory to save output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load stimulus
    print(f"Loading stimulus from: {stimulus_path}")
    stimulus, stimulus_sr = load_audio(Path(stimulus_path))

    # Ensure stimulus is mono
    stimulus = ensure_mono(stimulus)
    stimulus_name = Path(stimulus_path).stem

    # Expected RIR labels
    expected_labels = ['in-situ-near', 'in-situ-far', 'ex-situ-near', 'ex-situ-far']

    # Verify we have all required RIRs
    if not isinstance(rir_paths_dict, dict):
        raise ValueError("rir_paths_dict must be a dictionary")

    missing_labels = [label for label in expected_labels if label not in rir_paths_dict]
    if missing_labels:
        raise ValueError(f"Missing RIR labels: {missing_labels}")

    # Dictionary to store results for printing
    results = []

    print("\n" + "=" * 70)
    print("LOCALIZED STIMULUS CREATION")
    print("=" * 70)
    print(f"Input stimulus: {stimulus_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target RMS (loudspeaker): {LOUDSPEAKER_TARGET_RMS_DBFS} dBFS")
    print(f"Target RMS (spatial): {SPATIAL_TARGET_RMS_DBFS} dBFS")
    print(f"Bandpass filter: {BANDPASS_LOW_HZ} - {BANDPASS_HIGH_HZ} Hz")
    print(f"Silence padding: {SILENCE_PAD_SECONDS} s")
    print(f"Click ramp: {CLICK_RAMP_SECONDS} s")
    print(f"RMS preservation: {PRESERVE_RMS}")
    print(f"IR tail fade: {IR_TAIL_FADE_MS} ms" if IR_TAIL_FADE_MS else "IR tail fade: disabled")
    print("=" * 70 + "\n")

    # Load all RIRs first to calculate expected output length
    print("Loading RIRs...")
    rirs = {}
    rir_srs = {}
    for label in expected_labels:
        rir_path = rir_paths_dict[label]
        print(f"  Loading {label} from: {rir_path}")
        rir, rir_sr = load_rir_array(Path(rir_path), RIR_SAMPLE_RATE)
        rirs[label] = rir
        rir_srs[label] = rir_sr
        print(f"    Shape: {rir.shape}, SR: {rir_sr} Hz")
    print()

    # Calculate expected output length after convolution
    # Convolved length = stimulus_length + rir_length - 1
    # Use the first RIR to determine the expected length
    first_rir = rirs[expected_labels[0]]
    first_rir_sr = rir_srs[expected_labels[0]]

    # Account for potential resampling
    stimulus_for_convolution = stimulus.copy()
    if stimulus_sr != first_rir_sr:
        stimulus_for_convolution = resample_audio(stimulus_for_convolution, stimulus_sr, first_rir_sr)

    expected_convolved_length = stimulus_for_convolution.shape[0] + first_rir.shape[0] - 1

    # Pad stimulus with silence to match the convolved length
    # This ensures loudspeaker and spatial stimuli have the same duration
    silence_samples = expected_convolved_length - stimulus_for_convolution.shape[0]
    if silence_samples > 0:
        stimulus_padded = np.concatenate([stimulus_for_convolution, np.zeros(silence_samples, dtype=np.float32)])
        print(f"Padding stimulus with {silence_samples} samples ({silence_samples / first_rir_sr:.3f}s) to match convolved length")
        print(f"Original stimulus length: {stimulus_for_convolution.shape[0]} samples")
        print(f"Expected convolved length: {expected_convolved_length} samples\n")
    else:
        stimulus_padded = stimulus_for_convolution
        print(f"No padding needed - stimulus already as long as convolved output\n")

    # Use padded stimulus for processing (resample back to original if needed)
    if stimulus_sr != first_rir_sr:
        stimulus_for_loudspeaker = resample_audio(stimulus_padded, first_rir_sr, stimulus_sr)
    else:
        stimulus_for_loudspeaker = stimulus_padded

    # Process loudspeaker stimulus with padding
    print("Processing loudspeaker stimulus...")
    try:
        loudspeaker_audio = process_loudspeaker(stimulus_for_loudspeaker, stimulus_sr)
        loudspeaker_filename = f"{stimulus_name}_loudspeaker.wav"
        loudspeaker_filepath = output_path / loudspeaker_filename
        save_audio(loudspeaker_filepath, loudspeaker_audio, OUTPUT_SAMPLE_RATE)
        print(f"  Saved: {loudspeaker_filepath}\n")
        results.append({
            'filename': loudspeaker_filename,
            'rir_label': 'loudspeaker',
            'rir_path': 'N/A',
            'output_path': str(loudspeaker_filepath)
        })
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Process each RIR
    for label in expected_labels:
        rir = rirs[label]
        rir_sr = rir_srs[label]
        rir_path = rir_paths_dict[label]

        print(f"Processing {label}...")
        try:
            localized_audio = process_spatial(
                stimulus,  # Use original stimulus, not padded version
                stimulus_sr,
                rir,
                rir_sr,
                preserve_rms=PRESERVE_RMS,
                max_amp=MAX_PEAK,
                ir_tail_fade_duration_ms=IR_TAIL_FADE_MS,
            )

            output_filename = f"{stimulus_name}_{label}.wav"
            output_filepath = output_path / output_filename
            save_audio(output_filepath, localized_audio, OUTPUT_SAMPLE_RATE)
            print(f"  Saved: {output_filepath}\n")

            results.append({
                'filename': output_filename,
                'rir_label': label,
                'rir_path': rir_path,
                'output_path': str(output_filepath)
            })
        except Exception as e:
            print(f"  ERROR: {e}\n")

    # Print verification summary
    print("=" * 70)
    print("CREATION SUMMARY")
    print("=" * 70)
    print(f"Input stimulus: {stimulus_path}")
    print(f"Output directory: {output_dir}\n")

    for result in results:
        print(f"Generated file: {result['filename']}")
        print(f"  Type: {result['rir_label']}")
        if result['rir_path'] != 'N/A':
            print(f"  RIR source: {result['rir_path']}")
        print()

    print("=" * 70)
    print(f"Total files generated: {len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    create_localized_stimuli(STIMULUS_PATH, RIR_PATHS, output_dir=OUTPUT_DIR)
