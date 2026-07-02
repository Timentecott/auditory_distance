# -*- coding: utf-8 -*-
"""
Take as input a file path to a dry audio stimulus and file paths to four RIRs (.npy).
The four RIRs are: in-situ-near, in-situ-far, ex-situ-near, ex-situ-far.
Convolve the dry stimulus with each RIR and save the resulting localized audio files to disk 
(in a "recordings" folder). Files are saved with informative names (e.g., "stimulusname_in-situ-near.wav")
and normalized to -20 dBFS RMS. Prints generated filenames and corresponding RIRs for verification.
"""

import argparse
import os
import numpy as np
import scipy.signal
import soundfile as sf
from pathlib import Path


def compute_rms(x: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono by averaging channels if stereo, or return as-is if already mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)



def load_audio(file_path):
    """
    Load audio from .wav file.

    Args:
        file_path (str): Path to audio file (.wav)

    Returns:
        tuple: (audio_data, sample_rate)
    """
    file_path = str(file_path)

    audio, sr = sf.read(file_path)
    return audio, sr


def load_rir(rir_path):
    """
    Load an RIR from .npy file and reshape to (n_samples, n_channels).

    Args:
        rir_path (str): Path to RIR file (.npy)

    Returns:
        tuple: (rir_array, sample_rate) where rir_array has shape (n_samples, n_channels)
    """
    rir_path = str(rir_path)

    if not rir_path.endswith('.npy'):
        raise ValueError(f"Unsupported RIR format: {rir_path}. Use .npy")

    rir = np.load(rir_path)
    sr = 44100  # Default sample rate for RIRs

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


def normalize_audio(audio, max_amp=0.999):
    """Normalize audio to -20 dBFS RMS and prevent clipping.

    Args:
        audio (np.ndarray): Audio to normalize
        max_amp (float): Maximum allowed peak amplitude

    Returns:
        np.ndarray: Normalized audio
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Target RMS in linear scale: -20 dBFS = 10^(-20/20)
    target_rms_dbfs = -20.0
    target_rms_linear = 10.0 ** (target_rms_dbfs / 20.0)

    # Scale to target RMS
    current_rms = compute_rms(audio)
    if current_rms > 0:
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
        stimulus_path (str): Path to dry audio stimulus file (.wav)
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

    # Ensure stimulus is mono (average channels if stereo)
    stimulus = ensure_mono(stimulus)

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

        # Convolve with binaural RIR
        convolved = convolve_with_rir(stimulus, rir)

        # Generate output filename
        output_filename = f"{stimulus_name}_{label}.wav"
        output_filepath = output_path / output_filename

        # Normalize to -20 dBFS RMS and prevent clipping before saving
        convolved = normalize_audio(convolved, max_amp=0.999)

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
    parser = argparse.ArgumentParser(
        description="Convolve audio stimuli with four RIRs (in-situ-near, in-situ-far, ex-situ-near, ex-situ-far) and save localized audio files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to audio file to be localized (.wav)"
    )
    parser.add_argument(
        "--in-situ-near-rir",
        type=Path,
        required=True,
        help="Path to in-situ-near RIR file (.npy)"
    )
    parser.add_argument(
        "--in-situ-far-rir",
        type=Path,
        required=True,
        help="Path to in-situ-far RIR file (.npy)"
    )
    parser.add_argument(
        "--ex-situ-near-rir",
        type=Path,
        required=True,
        help="Path to ex-situ-near RIR file (.npy)"
    )
    parser.add_argument(
        "--ex-situ-far-rir",
        type=Path,
        required=True,
        help="Path to ex-situ-far RIR file (.npy)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\Localised"),
        help="Output directory for localized audio files (default: experiment_2/audio_stimuli/Localised)"
    )

    args = parser.parse_args()

    # Build the RIR dictionary from command-line arguments
    rir_dict = {
        'in-situ-near': args.in_situ_near_rir,
        'in-situ-far': args.in_situ_far_rir,
        'ex-situ-near': args.ex_situ_near_rir,
        'ex-situ-far': args.ex_situ_far_rir
    }

    # Run the convolution and localization
    localise_with_recorded_rir(args.input, rir_dict, output_dir=args.output)
