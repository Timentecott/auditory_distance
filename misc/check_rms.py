"""Check RMS values for one audio file or every audio file in a folder."""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pyfar as pf
import pyfar.dsp as dsp


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au"}


def a_weighting_db(frequencies_hz: np.ndarray) -> np.ndarray:
	"""Return A-weighting in dB for the given frequencies."""
	frequencies_hz = np.asarray(frequencies_hz, dtype=float)
	weighting_db = np.full_like(frequencies_hz, -np.inf, dtype=float)
	positive_frequencies = frequencies_hz > 0
	if not np.any(positive_frequencies):
		return weighting_db

	f = frequencies_hz[positive_frequencies]
	f2 = f * f
	term_1 = f2 + 20.6**2
	term_2 = f2 + 12200.0**2
	term_3 = np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
	a_db = 2.0 + 20.0 * np.log10((12200.0**2 * f2**2) / (term_1 * term_2 * term_3))
	weighting_db[positive_frequencies] = a_db
	return weighting_db


def apply_a_weighting(samples: np.ndarray, sampling_rate: float) -> np.ndarray:
	"""Apply A-weighting in the frequency domain and return weighted samples."""
	if samples.size == 0:
		return samples

	n_samples = samples.shape[-1]
	frequencies_hz = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
	weighting_linear = 10.0 ** (a_weighting_db(frequencies_hz) / 20.0)
	weighted_spectrum = np.fft.rfft(samples, axis=-1) * weighting_linear
	return np.fft.irfft(weighted_spectrum, n=n_samples, axis=-1)


def average_lr_rms(signal: pf.Signal) -> float:
	"""Return the mean of left/right RMS values."""
	samples = np.asarray(signal.time)
	if samples.ndim == 1:
		return float(dsp.rms(signal))
	if samples.shape[0] == 1:
		return float(dsp.rms(signal))
	channel_rms = np.asarray(dsp.rms(signal)).reshape(-1)
	if channel_rms.size == 1:
		return float(channel_rms[0])
	return float(np.mean(channel_rms[:2]))


def find_audio_files(path: Path) -> list[Path]:
	if path.is_file():
		return [path]
	if path.is_dir():
		return sorted(
			candidate for candidate in path.rglob("*")
			if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
		)
	raise FileNotFoundError(f"Path does not exist: {path}")


def format_rms(value) -> str:
	array = np.asarray(value)
	if array.ndim == 0:
		return f"{float(array):.6f}"
	return np.array2string(array, precision=6, separator=", ")


def rms_to_dbfs(value) -> str:
	array = np.asarray(value, dtype=float)
	with np.errstate(divide="ignore"):
		dbfs = 20.0 * np.log10(array)
	if dbfs.ndim == 0:
		return f"{float(dbfs):.2f}"
	return np.array2string(dbfs, precision=2, separator=", ")


def to_plot_scalar(value) -> float:
	array = np.asarray(value, dtype=float)
	if array.ndim == 0:
		return float(array)
	return float(np.mean(array))


def show_summary_plot(results: list[dict]) -> None:
	file_labels = [result["label"] for result in results]
	rms_dbfs_values = [result["rms_dbfs_plot"] for result in results]
	a_weighted_dbfs_values = [result["a_weighted_dbfs"] for result in results]
	lr_average_dbfs_values = [result["lr_average_dbfs"] for result in results]
	predicted_spl_values = [result["predicted_spl_plot"] for result in results]

	fig, axes = plt.subplots(2, 1, figsize=(max(10, len(results) * 1.2), 8), sharex=True)
	fig.suptitle("RMS summary by file")

	axes[0].plot(file_labels, rms_dbfs_values, marker="o", label="RMS dBFS")
	axes[0].plot(file_labels, a_weighted_dbfs_values, marker="o", label="A-weighted dBFS")
	axes[0].plot(file_labels, lr_average_dbfs_values, marker="o", label="L/R average dBFS")
	axes[0].set_ylabel("dBFS")
	axes[0].grid(True, alpha=0.3)
	axes[0].legend()

	axes[1].plot(file_labels, predicted_spl_values, marker="o", color="tab:green", label="Predicted SPL")
	axes[1].set_ylabel("dB SPL")
	axes[1].set_xlabel("Audio file")
	axes[1].grid(True, alpha=0.3)
	axes[1].legend()

	for axis in axes:
		axis.tick_params(axis="x", rotation=45)

	fig.tight_layout()
	plt.show()
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Print RMS values for one audio file or all audio files in a folder."
	)
	parser.add_argument(
		"path",
		nargs="?",
		default=r"E:\PhD_experiments\auditory_distance\experiment_1\localised_stimuli_B20\ISTS\ISTS-V1.0_60s_24bit_4_BRIR_localised_BRIRs_from_a_room_B_020.wav",
		help="Audio file or folder to analyse.",
	)
	parser.add_argument(
		"--calibration-offset-db",
		type=float,
		default=0.0,
		help="Add this offset in dB to convert dBFS to predicted SPL.",
	)
	args = parser.parse_args()

	target_path = Path(args.path)
	audio_files = find_audio_files(target_path)
	results = []

	for audio_path in audio_files:
		signal = pf.io.read_audio(audio_path)
		rms_value = dsp.rms(signal)
		weighted_samples = apply_a_weighting(np.asarray(signal.time), signal.sampling_rate)
		a_weighted_rms = float(np.sqrt(np.mean(weighted_samples**2)))
		lr_average_rms = average_lr_rms(signal)
		rms_dbfs = rms_to_dbfs(rms_value)
		a_weighted_dbfs = 20.0 * np.log10(a_weighted_rms) if a_weighted_rms > 0 else float("-inf")
		lr_average_dbfs = 20.0 * np.log10(lr_average_rms) if lr_average_rms > 0 else float("-inf")
		predicted_spl = np.asarray(rms_value, dtype=float)
		with np.errstate(divide="ignore"):
			predicted_spl = 20.0 * np.log10(predicted_spl) + args.calibration_offset_db
		predicted_spl_plot = to_plot_scalar(predicted_spl)
		rms_dbfs_plot = to_plot_scalar(np.asarray(rms_value, dtype=float))
		print(
			f"{audio_path}\tRMS: {format_rms(rms_value)}"
			f"\tRMS dBFS: {rms_dbfs}"
			f"\tPredicted SPL: {np.array2string(np.asarray(predicted_spl), precision=2, separator=', ')} dB SPL"
			f"\tA-weighted RMS: {a_weighted_rms:.6f}"
			f"\tA-weighted dBFS: {a_weighted_dbfs:.2f}"
			f"\tL/R average RMS: {lr_average_rms:.6f}"
			f"\tL/R average dBFS: {lr_average_dbfs:.2f}"
		)
		results.append({
			"label": audio_path.stem,
			"rms_dbfs_plot": rms_dbfs_plot,
			"a_weighted_dbfs": a_weighted_dbfs,
			"lr_average_dbfs": lr_average_dbfs,
			"predicted_spl_plot": predicted_spl_plot,
		})

	if results:
		show_summary_plot(results)


if __name__ == "__main__":
	main()
