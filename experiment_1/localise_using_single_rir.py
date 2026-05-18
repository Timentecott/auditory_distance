#!/usr/bin/env python3
"""Localise all audio files in an input folder by convolving them with a single RIR.

This script mirrors the typical workflow used by tools that apply an RIR to dry audio:
- Load a single RIR (NumPy .npy file, shape (n_samples,) or (n_samples, n_channels)).
- For each audio file found under the input folder, optionally resample to the RIR sample rate,
  create a mono source (mix if necessary), convolve with each RIR channel, and save a multi-channel
  localized output file into the output folder while preserving directory structure.

Examples:
    python localise_using_single_rir.py \
        --input "C:/path/to/dry_wavs" \
        --output "C:/path/to/out" \
        --rir "C:/path/to/RIR.npy" \
        --rir-sr 48000

The script will create the same subfolder layout in the output directory and save files with
identical names. By default the RIR sample rate is assumed to be 48000 Hz; pass --rir-sr to change.

"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import math
import sys

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".mp3"}


def find_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    raise FileNotFoundError(f"Path does not exist: {path}")


def load_rir(rir_path: Path) -> np.ndarray:
    """Load an RIR saved as a numpy .npy file.

    Returns an array with shape (n_samples, n_channels).
    """
    arr = np.load(str(rir_path))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2:
        # keep (n_samples, n_channels)
        pass
    else:
        raise ValueError("RIR array must have 1 or 2 dimensions")
    return arr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio (shape (n,) or (n, ch)) to target_sr using FFT resampling.

    Uses scipy.signal.resample which may be sufficient for this use-case.
    """
    if orig_sr == target_sr:
        return audio
    ratio = float(target_sr) / float(orig_sr)
    n_samples = int(round(audio.shape[0] * ratio))
    if audio.ndim == 1:
        return signal.resample(audio, n_samples)
    else:
        # apply resample per channel
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(signal.resample(audio[:, ch], n_samples))
        return np.column_stack(channels)


def compute_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def convolve_with_rir(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve a mono source (1D array) with an RIR array (n_samples, n_channels).

    Returns an array shaped (n_out_samples, n_channels).
    """
    if source.ndim != 1:
        raise ValueError("Source must be 1-D mono array for convolution")
    n_ch = rir.shape[1]
    out_len = source.shape[0] + rir.shape[0] - 1
    out = np.zeros((out_len, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        out[:, ch] = signal.fftconvolve(source, rir[:, ch], mode='full')
    return out


def make_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_file.is_file():
        rel = input_file.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    raise ValueError("input_file must be a file")


def process_file(in_path: Path, out_path: Path, rir: np.ndarray, rir_sr: int, preserve_rms=True, max_amp=0.999):
    audio, sr = sf.read(str(in_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    # mix to mono if necessary
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # resample if needed
    if sr != rir_sr:
        mono = resample_audio(mono, orig_sr=sr, target_sr=rir_sr)
        sr = rir_sr

    orig_rms = compute_rms(mono)

    # convolve
    localized = convolve_with_rir(mono, rir)

    # preserve RMS (optional)
    if preserve_rms and orig_rms > 0:
        new_rms = compute_rms(localized)
        if new_rms > 0:
            scale = orig_rms / new_rms
            localized = localized * scale

    # avoid clipping
    peak = float(np.max(np.abs(localized))) if localized.size else 0.0
    if peak > max_amp:
        localized = localized * (max_amp / peak)

    # write file - use rir_sr as sample rate
    sf.write(str(out_path), localized, rir_sr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Localise dry audio files using a single RIR (NumPy .npy) and save outputs.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input file or folder containing dry audio files")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output folder to write localised audio (keeps subfolders)")
    parser.add_argument("--rir", "-r", required=True, type=Path, help="Path to RIR .npy file (array shape (n_samples,) or (n_samples, n_channels))")
    parser.add_argument("--rir-sr", type=int, default=48000, help="Sample rate assumed for the RIR (Hz). Audio will be resampled to this rate if needed")
    parser.add_argument("--preserve-rms", action='store_true', help="Scale localized output to preserve input RMS (default: False)")
    parser.add_argument("--max-amp", type=float, default=0.999, help="Maximum allowed peak amplitude to avoid clipping (default 0.999)")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing output files")
    args = parser.parse_args(argv)

    input_root: Path = args.input
    output_root: Path = args.output
    rir_path: Path = args.rir
    rir_sr = int(args.rir_sr)

    if not input_root.exists():
        print(f"Input path not found: {input_root}")
        return 2

    # If the user passed a directory for --rir, look for .npy files inside.
    if not rir_path.exists():
        print(f"RIR file not found: {rir_path}")
        return 2
    if rir_path.is_dir():
        npy_files = sorted(p for p in rir_path.iterdir() if p.is_file() and p.suffix.lower() == '.npy')
        if len(npy_files) == 0:
            print(f"No .npy RIR files found in directory: {rir_path}")
            return 2
        elif len(npy_files) == 1:
            print(f"Using RIR file from directory: {npy_files[0]}")
            rir_path = npy_files[0]
        else:
            print(f"Multiple .npy files found in RIR directory {rir_path}. Please specify a single .npy file. Found:")
            for p in npy_files:
                print(f"  {p}")
            return 2

    print(f"Loading RIR from: {rir_path}")
    rir = load_rir(rir_path)
    print(f"RIR shape: {rir.shape}, assumed sample rate: {rir_sr} Hz")

    files = find_audio_files(input_root)
    if not files:
        print(f"No audio files found in: {input_root}")
        return 1

    print(f"Found {len(files)} audio files")
    for idx, in_file in enumerate(files, start=1):
        rel = in_file.relative_to(input_root)
        out_file = output_root / rel
        out_file = out_file.with_suffix('.wav')
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists() and not args.overwrite:
            print(f"Skipping existing file: {out_file}")
            continue

        try:
            print(f"[{idx}/{len(files)}] Processing: {in_file} -> {out_file}")
            process_file(in_file, out_file, rir, rir_sr, preserve_rms=args.preserve_rms, max_amp=args.max_amp)
        except Exception as e:
            print(f"  ERROR processing {in_file}: {e}")

    print("Done")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


