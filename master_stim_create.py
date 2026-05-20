#!/usr/bin/env python3
"""Create loudspeaker, in_situ, and ex_situ stimulus sets from dry source audio.

This script:
- Recursively reads all audio files from the input folder.
- Creates loudspeaker stimuli by converting audio to mono, routing it to the left channel,
  and normalizing to target RMS 0.1.
- Creates in_situ and ex_situ stimuli using the same single-RIR convolution workflow as
  localise_using_single_rir.py: mono mix, optional resampling, per-channel FFT convolution,
  optional RMS preservation, and peak limiting.
- Preserves the input directory structure in each output folder.C:

Examples:
    python master_stim_create.py
    python master_stim_create.py --input "C:/path/to/original_audios"
    python master_stim_create.py --in-situ-rir "C:/path/to/RIR.npy" --ex-situ-rir "C:/path/to/RIR.npy"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".mp3"}
DEFAULT_RIR_SR = 48000
LOUDSPEAKER_TARGET_RMS = 0.1
SPATIAL_TARGET_RMS = 0.05
MAX_PEAK = 0.999


def find_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
        )
    raise FileNotFoundError(f"Path does not exist: {path}")


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


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
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def mono_left_only_stereo(audio: np.ndarray) -> np.ndarray:
    mono = ensure_mono(audio)
    stereo = np.zeros((mono.shape[0], 2), dtype=np.float32)
    stereo[:, 0] = mono.astype(np.float32)
    return stereo


def normalize_audio(audio: np.ndarray, target_rms: float, max_peak: float = MAX_PEAK) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    current_rms = compute_rms(audio)
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > max_peak:
        audio = audio * (max_peak / peak)
    return audio.astype(np.float32)


def load_rir_array(rir_path: Path, rir_sr: int) -> tuple[np.ndarray, int]:
    """Load an RIR/BRIR from .npy or .sofa.

    Returns a 2D array with shape (n_samples, n_channels) and an associated sample rate.
    """
    suffix = rir_path.suffix.lower()

    if suffix == ".npy":
        rir = np.load(str(rir_path))
        rir = np.asarray(rir, dtype=np.float32)
        if rir.ndim == 1:
            rir = rir[:, None]
        elif rir.ndim != 2:
            raise ValueError(f"RIR array must be 1D or 2D, got shape {rir.shape}")
        return rir, int(rir_sr)

    if suffix == ".sofa":
        try:
            import sofar as sofa
        except Exception as exc:
            raise ImportError("Reading .sofa files requires the 'sofar' package.") from exc

        sofa_data = sofa.read_sofa(str(rir_path), verify=False)
        ir = np.asarray(sofa_data.Data_IR, dtype=np.float32)
        if ir.ndim == 3:
            ir = ir[0]
        if ir.ndim == 1:
            ir = ir[:, None]
        elif ir.ndim != 2:
            raise ValueError(f"Unexpected SOFA IR shape: {ir.shape}")

        sr_values = getattr(sofa_data, "Data_SamplingRate", None)
        if sr_values is None:
            sr = rir_sr
        else:
            sr = int(np.asarray(sr_values).ravel()[0])
        return ir, sr

    raise ValueError(f"Unsupported RIR format: {rir_path.suffix} (use .npy or .sofa)")


def convolve_with_rir(source_mono: np.ndarray, rir: np.ndarray) -> np.ndarray:
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


def make_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    relative = input_file.relative_to(input_root)
    out_path = output_root / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.with_suffix(".wav")


def process_loudspeaker(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = ensure_mono(audio)
    stereo = mono_left_only_stereo(mono)
    return normalize_audio(stereo, LOUDSPEAKER_TARGET_RMS)


def process_spatial(
    audio: np.ndarray,
    sample_rate: int,
    rir: np.ndarray,
    rir_sr: int,
    preserve_rms: bool = False,
    max_amp: float = MAX_PEAK,
) -> np.ndarray:
    """Mirror localise_using_single_rir.py processing, then normalize to a spatial RMS target."""
    mono = ensure_mono(audio)
    if sample_rate != rir_sr:
        mono = resample_audio(mono, sample_rate, rir_sr)
        sample_rate = rir_sr

    orig_rms = compute_rms(mono)
    localized = convolve_with_rir(mono, rir)

    if preserve_rms and orig_rms > 0:
        new_rms = compute_rms(localized)
        if new_rms > 0:
            localized = localized * (orig_rms / new_rms)

    localized = normalize_audio(localized, SPATIAL_TARGET_RMS, max_peak=max_amp)
    return localized.astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def process_folder(
    input_root: Path,
    output_root: Path,
    rir: np.ndarray | None = None,
    rir_sr: int = DEFAULT_RIR_SR,
    overwrite: bool = False,
    preserve_rms: bool = False,
    max_amp: float = MAX_PEAK,
) -> int:
    files = find_audio_files(input_root)
    if not files:
        print(f"No audio files found in {input_root}")
        return 0

    processed = 0
    print(f"Found {len(files)} audio files in {input_root}")

    for idx, audio_path in enumerate(files, start=1):
        out_path = make_output_path(audio_path, input_root, output_root)
        if out_path.exists() and not overwrite:
            print(f"Skipping existing file: {out_path}")
            continue

        try:
            audio, sample_rate = load_audio(audio_path)
            if rir is None:
                processed_audio = process_loudspeaker(audio, sample_rate)
                output_sr = sample_rate
            else:
                processed_audio = process_spatial(
                    audio,
                    sample_rate,
                    rir,
                    rir_sr,
                    preserve_rms=preserve_rms,
                    max_amp=max_amp,
                )
                output_sr = rir_sr

            save_audio(out_path, processed_audio, output_sr)
            processed += 1
            print(f"[{idx}/{len(files)}] {audio_path} -> {out_path}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] ERROR processing {audio_path}: {exc}")

    return processed


def build_default_paths(repo_root: Path) -> tuple[Path, Path, Path, Path]:
    experiment_root = repo_root / "experiment_1"
    input_root = experiment_root / "original_audios"
    loudspeaker_root = experiment_root / "loudspeaker_stimuli_bib"
    in_situ_root = experiment_root / "in_situ_stimuli_bib"
    ex_situ_root = experiment_root / "ex_situ_stimuli_bib"
    return input_root, loudspeaker_root, in_situ_root, ex_situ_root


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create loudspeaker, in_situ, and ex_situ stimulus folders from dry audio."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input folder containing original audio files (default: experiment_1/original_audios).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Base output folder (default: experiment_1).",
    )
    parser.add_argument(
        "--in-situ-rir",
        type=Path,
        default=None,
        help="Path to in_situ RIR/BRIR (.npy or .sofa).",
    )
    parser.add_argument(
        "--ex-situ-rir",
        type=Path,
        default=None,
        help="Path to ex_situ RIR/BRIR (.npy or .sofa).",
    )
    parser.add_argument(
        "--rir-sr",
        type=int,
        default=DEFAULT_RIR_SR,
        help="Sample rate to assume for .npy RIR files (default: 48000).",
    )
    parser.add_argument(
        "--preserve-rms",
        action="store_true",
        help="Preserve input RMS after spatial convolution, matching localise_using_single_rir.py's optional mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent
    default_input_root, default_loudspeaker_root, default_in_situ_root, default_ex_situ_root = build_default_paths(repo_root)

    input_root = args.input or default_input_root
    output_base = args.output_base or (repo_root / "experiment_1")
    loudspeaker_root = default_loudspeaker_root if args.output_base is None else output_base / "loudspeaker_stimuli_bob"
    in_situ_root = default_in_situ_root if args.output_base is None else output_base / "in_situ_stimuli_bob"
    ex_situ_root = default_ex_situ_root if args.output_base is None else output_base / "ex_situ_stimuli_bob"

    in_situ_rir_path = args.in_situ_rir or (repo_root / "experiment_1" / "resources" / "tim_lab_headphoneRIR.npy")
    ex_situ_rir_path = args.ex_situ_rir or (repo_root / "experiment_1" / "resources" / "tim_otherlabRIR.npy")

    print("=" * 70)
    print("MASTER STIMULUS CREATION")
    print("=" * 70)
    print(f"Input folder: {input_root}")
    print(f"Loudspeaker output: {loudspeaker_root}")
    print(f"In-situ output: {in_situ_root}")
    print(f"Ex-situ output: {ex_situ_root}")
    print(f"In-situ RIR: {in_situ_rir_path}")
    print(f"Ex-situ RIR: {ex_situ_rir_path}")
    print(f"Spatial RMS preservation: {args.preserve_rms}")
    print("=" * 70)

    if not input_root.exists():
        print(f"Input folder not found: {input_root}")
        return 2

    loudspeaker_root.mkdir(parents=True, exist_ok=True)
    in_situ_root.mkdir(parents=True, exist_ok=True)
    ex_situ_root.mkdir(parents=True, exist_ok=True)

    in_situ_rir, in_situ_rir_sr = load_rir_array(in_situ_rir_path, args.rir_sr)
    ex_situ_rir, ex_situ_rir_sr = load_rir_array(ex_situ_rir_path, args.rir_sr)

    print("\nCreating loudspeaker stimuli...")
    loudspeaker_count = process_folder(
        input_root=input_root,
        output_root=loudspeaker_root,
        rir=None,
        overwrite=args.overwrite,
    )

    print("\nCreating in_situ stimuli...")
    in_situ_count = process_folder(
        input_root=input_root,
        output_root=in_situ_root,
        rir=in_situ_rir,
        rir_sr=in_situ_rir_sr,
        overwrite=args.overwrite,
        preserve_rms=args.preserve_rms,
    )

    print("\nCreating ex_situ stimuli...")
    ex_situ_count = process_folder(
        input_root=input_root,
        output_root=ex_situ_root,
        rir=ex_situ_rir,
        rir_sr=ex_situ_rir_sr,
        overwrite=args.overwrite,
        preserve_rms=args.preserve_rms,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Loudspeaker files written: {loudspeaker_count}")
    print(f"In-situ files written: {in_situ_count}")
    print(f"Ex-situ files written: {ex_situ_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
