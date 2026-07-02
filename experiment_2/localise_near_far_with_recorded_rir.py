# -*- coding: utf-8 -*-
"""
Take as input a file path to a dry audio stimulus and file paths to four RIRs (either .npy or .wav).
The four RIRs are: in-situ-near, in-situ-far, ex-situ-near, ex-situ-far.
Convolve the dry stimulus with each RIR and save the resulting localized audio files to disk 
(in a "recordings" folder). Files are saved with informative names (e.g., "stimulusname_in-situ-near.wav")
and normalized to prevent clipping. Prints generated filenames and corresponding RIRs for verification.
"""

import os
import numpy as np
import scipy.signal
import soundfile as sf
from pathlib import Path


def compute_rms(x: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio (shape (n,) or (n, ch)) to target_sr using FFT resampling.

    Args:
        audio: Audio array, 1D or 2D
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    ratio = float(target_sr) / float(orig_sr)
    n_samples = int(round(audio.shape[0] * ratio))
    if audio.ndim == 1:
        return scipy.signal.resample(audio, n_samples)
    else:
        # apply resample per channel
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(scipy.signal.resample(audio[:, ch], n_samples))
        return np.column_stack(channels)


def load_audio(file_path):
    """
    Load audio from either .wav or .npy file.

    Args:
        file_path (str): Path to audio file (.wav or .npy)

    Returns:
        tuple: (audio_data, sample_rate)
    """
    file_path = str(file_path)

    if file_path.endswith('.wav'):
        audio, sr = sf.read(file_path)
        return audio, sr
    elif file_path.endswith('.npy'):
        audio = np.load(file_path)
        sr = 44100  # Default sample rate for .npy files (can be adjusted)
        return audio, sr
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .wav or .npy")


def load_rir(rir_path):
    """
    Load an RIR from .npy or .wav file and reshape to (n_samples, n_channels).

    Args:
        rir_path (str): Path to RIR file (.npy or .wav)

    Returns:
        tuple: (rir_array, sample_rate) where rir_array has shape (n_samples, n_channels)
    """
    rir_path = str(rir_path)

    if rir_path.endswith('.wav'):
        rir, sr = sf.read(rir_path)
    elif rir_path.endswith('.npy'):
        rir = np.load(rir_path)
        sr = 44100
    else:
        raise ValueError(f"Unsupported RIR format: {rir_path}. Use .wav or .npy")

    rir = np.asarray(rir, dtype=np.float32)

    # Reshape 1D to (n_samples, 1) if needed
    if rir.ndim == 1:
        rir = rir[:, None]
    elif rir.ndim == 2:
        pass  # Already (n_samples, n_channels)
    else:
        raise ValueError("RIR array must have 1 or 2 dimensions")

    return rir, sr


def convolve_with_rir(source, rir):
    """Convolve a mono source (1D array) with an RIR array (n_samples, n_channels).

    Args:
        source (np.ndarray): Mono audio source, shape (n_samples,)
        rir (np.ndarray): Room impulse response, shape (n_samples, n_channels)

    Returns:
        np.ndarray: Convolved audio with shape (n_out_samples, n_channels)
    """
    if source.ndim != 1:
        raise ValueError("Source must be 1-D mono array for convolution")
    n_ch = rir.shape[1]
    out_len = source.shape[0] + rir.shape[0] - 1
    out = np.zeros((out_len, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        out[:, ch] = scipy.signal.fftconvolve(source, rir[:, ch], mode='full')
    return out


def normalize_audio(audio, target_rms_db=None, max_amp=0.999):
    """Normalize audio to prevent clipping and target specific RMS level.

    Args:
        audio (np.ndarray): Audio to normalize
        target_rms_db (float): Target RMS level in dB (e.g., -12.0 for -12dB). If None, only prevents clipping.
        max_amp (float): Maximum allowed peak amplitude

    Returns:
        np.ndarray: Normalized audio
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Scale to target RMS if specified
    if target_rms_db is not None:
        current_rms = compute_rms(audio)
        if current_rms > 0:
            # Convert dB to linear: linear = 10^(dB/20)
            target_rms_linear = 10.0 ** (target_rms_db / 20.0)
            scale = target_rms_linear / current_rms
            audio = audio * scale

    # Avoid clipping
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > max_amp:
        audio = audio * (max_amp / peak)

    return audio


def localise_with_recorded_rir(stimulus_path, rir_paths_dict, output_dir='recordings'):
    """
    Convolve a dry stimulus with multiple RIRs and save localized audio files.

    Args:
        stimulus_path (str): Path to dry audio stimulus file (.wav or .npy)
        rir_paths_dict (dict): Dictionary with keys as labels and values as RIR file paths
                              Keys should be: 'in-situ-near', 'in-situ-far', 'ex-situ-near', 'ex-situ-far'
        output_dir (str): Directory to save output files (default: 'recordings')
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load stimulus
    print(f"Loading stimulus from: {stimulus_path}")
    stimulus, stimulus_sr = load_audio(stimulus_path)

    # Ensure stimulus is 1D (take first channel if stereo)
    if stimulus.ndim > 1:
        stimulus = stimulus[:, 0]

    # Get stimulus filename without extension
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

    # Process each RIR
    for label in expected_labels:
        rir_path = rir_paths_dict[label]
        print(f"\nProcessing {label}...")
        print(f"  Loading RIR from: {rir_path}")

        # Load RIR (will be reshaped to (n_samples, n_channels))
        rir, rir_sr = load_rir(rir_path)
        print(f"  RIR shape: {rir.shape}")

        # Resample RIR if necessary
        if rir_sr != stimulus_sr:
            print(f"  Resampling RIR from {rir_sr} Hz to {stimulus_sr} Hz")
            rir = resample_audio(rir, orig_sr=rir_sr, target_sr=stimulus_sr)

        # Convolve with binaural RIR
        convolved = convolve_with_rir(stimulus, rir)

        # Normalize to -12dB RMS and prevent clipping
        convolved = normalize_audio(convolved, target_rms_db=-12.0, max_amp=0.999)

        # Generate output filename
        output_filename = f"{stimulus_name}_{label}.wav"
        output_filepath = output_path / output_filename

        # Save to disk (convolved is now multichannel)
        sf.write(str(output_filepath), convolved, stimulus_sr)
        print(f"  Saved: {output_filepath}")

        # Store result for summary
        results.append({
            'filename': output_filename,
            'rir_label': label,
            'rir_path': rir_path,
            'output_path': str(output_filepath)
        })

    # Print verification summary
    print("\n" + "="*60)
    print("CONVOLUTION SUMMARY")
    print("="*60)
    print(f"Input stimulus: {stimulus_path}")
    print(f"Output directory: {output_dir}\n")

    for result in results:
        print(f"Generated file: {result['filename']}")
        print(f"  RIR type: {result['rir_label']}")
        print(f"  RIR source: {result['rir_path']}")
        print()

    print("="*60)
    print(f"Total files generated: {len(results)}")
    print("="*60)


if __name__ == "__main__":
    # Example usage:
    # Specify the path to your dry stimulus
    stimulus_file = r"C:\Users\tim_e\source\repos\auditory_distance\posner\audio_stimuli\pink_noise_48k_30s_300_8000hz.wav"

    # Specify paths to your four RIRs
    rir_dict = {
        'in-situ-near': r"C:\Users\tim_e\source\repos\auditory_distance\posner\BRIR\near_lab_22_6\RIR.npy",
        'in-situ-far': r"C:\Users\tim_e\source\repos\auditory_distance\posner\BRIR\far_lab_22_6\RIR.npy",
        'ex-situ-near': r"C:\Users\tim_e\source\repos\auditory_distance\posner\BRIR\near_classroom_22_6\RIR.npy",
        'ex-situ-far': r"C:\Users\tim_e\source\repos\auditory_distance\posner\BRIR\far_classroom_22_6\RIR.npy"
    }

    # Run the convolution and localization
    localise_with_recorded_rir(stimulus_file, rir_dict, output_dir=r"C:\Users\tim_e\source\repos\auditory_distance\posner\audio_stimuli\Localised")
