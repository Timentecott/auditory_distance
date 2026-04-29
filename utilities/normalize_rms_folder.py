"""Normalize audio files in a folder to a target RMS while preserving channel balance."""

from __future__ import annotations

from pathlib import Path
import argparse

import librosa
import numpy as np
import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au"}


def find_audio_files(path: Path) -> list[Path]:
	if path.is_file():
		return [path]
	if path.is_dir():
		return sorted(
			candidate for candidate in path.rglob("*")
			if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
		)
	raise FileNotFoundError(f"Path does not exist: {path}")


def load_audio(audio_path: Path) -> tuple[np.ndarray, int, sf.SoundFile]:
	"""Load audio with channel-first layout and preserve file metadata."""
	info = sf.info(str(audio_path))
	audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
	return np.asarray(audio, dtype=np.float32), sample_rate, info


def channel_first_to_soundfile_layout(audio: np.ndarray) -> np.ndarray:
	if audio.ndim == 1:
		return audio
	return np.transpose(audio)


def compute_rms(audio: np.ndarray) -> float:
	return float(np.sqrt(np.mean(np.asarray(audio, dtype=float) ** 2)))


def build_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
	if input_path.is_file():
		output_root.mkdir(parents=True, exist_ok=True)
		return output_root / input_path.name

	relative_path = input_path.relative_to(input_root)
	return output_root / relative_path


def normalize_audio_file(
	input_path: Path,
	output_path: Path,
	target_rms: float,
) -> tuple[float, float, float]:
	audio, sample_rate, info = load_audio(input_path)
	current_rms = compute_rms(audio)
	if current_rms <= 0:
		raise ValueError(f"Cannot normalize silent audio file: {input_path}")

	gain = target_rms / current_rms
	normalized_audio = audio * gain
	output_path.parent.mkdir(parents=True, exist_ok=True)
	sf.write(
		str(output_path),
		channel_first_to_soundfile_layout(normalized_audio),
		sample_rate,
		subtype=info.subtype,
	)
	return current_rms, target_rms, gain


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Normalize all audio files in a folder to a target RMS while preserving stereo balance."
	)
	parser.add_argument(
		"path",
		help="Input audio file or folder to process.",
	)
	parser.add_argument(
		"--target-rms",
		type=float,
		required=True,
		help="Target RMS in linear full-scale units, for example 0.1 for -20 dBFS.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for normalized copies. Defaults to a sibling folder named normalized_rms.",
	)
	args = parser.parse_args()

	input_path = Path(args.path)
	audio_files = find_audio_files(input_path)

	if input_path.is_file():
		input_root = input_path.parent
		output_root = args.output_dir or input_path.parent / "normalized_rms"
	else:
		input_root = input_path
		output_root = args.output_dir or input_path.parent / f"{input_path.name}_normalized_rms"

	for audio_path in audio_files:
		output_path = build_output_path(audio_path, input_root, output_root)
		current_rms, target_rms, gain = normalize_audio_file(audio_path, output_path, args.target_rms)
		print(
			f"{audio_path} -> {output_path} | "
			f"current RMS={current_rms:.6f} | target RMS={target_rms:.6f} | gain={gain:.6f}"
		)


if __name__ == "__main__":
	main()