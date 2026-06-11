from __future__ import annotations

from pathlib import Path

import numpy as np
import pyfar as pf
import soundfile as sf
from scipy import signal


BASE_DIR = Path(__file__).resolve().parent
INPUT_AUDIO = BASE_DIR / "audio_stimuli" / "pink_noise_48k_30s_300_8000hz.wav"
OUTPUT_DIR = INPUT_AUDIO.parent

INPUT_DURATION_SEC = 0.5
OUTPUT_DURATION_SEC = 1.0
RIR_SAMPLE_RATE = 48000

BRIR_PATHS = {
    "near_left": BASE_DIR / "BRIR" / "home_near_left" / "RIR.npy",
    "near_right": BASE_DIR / "BRIR" / "home_near_right" / "RIR.npy",
    "far_left": BASE_DIR / "BRIR" / "home_far_left" / "RIR.npy",
    "far_right": BASE_DIR / "BRIR" / "home_far_right" / "RIR.npy",
}


def load_mono_signal(audio_path: Path) -> pf.Signal:
    audio, fs_audio = sf.read(str(audio_path))
    audio = np.asarray(audio)
    if audio.ndim == 1:
        mono = audio[np.newaxis, :]
    else:
        mono = np.mean(audio, axis=1, keepdims=False)[np.newaxis, :]
    return pf.Signal(mono, sampling_rate=fs_audio)


def trim_input_duration(audio_pf: pf.Signal, duration_sec: float) -> pf.Signal:
    target_samples = max(1, int(round(duration_sec * audio_pf.sampling_rate)))
    audio_time = np.asarray(audio_pf.time)
    trimmed = audio_time[..., :target_samples]
    if trimmed.shape[-1] < target_samples:
        print(
            f"WARNING: Input shorter than requested {duration_sec:.3f}s; "
            f"using {trimmed.shape[-1] / audio_pf.sampling_rate:.3f}s"
        )
    return pf.Signal(trimmed, sampling_rate=audio_pf.sampling_rate)


def load_rir(rir_path: Path) -> np.ndarray:
    rir = np.load(str(rir_path))
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim == 1:
        rir = rir[:, None]
    elif rir.ndim != 2:
        raise ValueError(f"RIR array must have 1 or 2 dimensions, got {rir.ndim}")
    return rir


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = float(target_sr) / float(orig_sr)
    n_samples = int(round(audio.shape[-1] * ratio))
    if audio.ndim == 1:
        return signal.resample(audio, n_samples)
    return np.vstack([signal.resample(ch, n_samples) for ch in audio])


def compute_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def convolve_with_rir(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if source.ndim != 1:
        raise ValueError("Source must be mono for convolution")
    out_len = source.shape[0] + rir.shape[0] - 1
    out = np.zeros((out_len, rir.shape[1]), dtype=np.float32)
    for ch in range(rir.shape[1]):
        out[:, ch] = signal.fftconvolve(source, rir[:, ch], mode="full")
    return out


def to_stereo_frames(signal_pf: pf.Signal) -> np.ndarray:
    t = np.asarray(signal_pf.time)
    t = np.squeeze(t)

    if t.ndim == 2 and t.shape[0] == 2:
        return t.T
    if t.ndim == 2 and t.shape[1] == 2:
        return t
    if t.ndim == 3:
        if t.shape[0] == 2:
            return t[:, :, 0].T
        if t.shape[1] == 2:
            return t[:, :2, 0]
        if t.shape[2] == 2:
            return t[:, 0, :]

    raise ValueError(f"Could not extract stereo channels from convolved signal with shape {t.shape}")


def save_binaural(output_path: Path, binaural: pf.Signal, output_duration_sec: float = None):
    stereo_frames = to_stereo_frames(binaural)
    if stereo_frames.ndim != 2 or stereo_frames.shape[1] != 2:
        raise ValueError(f"Output is not stereo. Shape: {stereo_frames.shape}")

    if output_duration_sec is not None:
        target_samples = max(1, int(round(output_duration_sec * binaural.sampling_rate)))
        if stereo_frames.shape[0] < target_samples:
            stereo_frames = np.pad(stereo_frames, ((0, target_samples - stereo_frames.shape[0]), (0, 0)), mode="constant")
        else:
            stereo_frames = stereo_frames[:target_samples, :]

    sf.write(str(output_path), stereo_frames, int(binaural.sampling_rate))
    print(f"Saved: {output_path} (shape={stereo_frames.shape})")


def localise_with_rir(audio_pf: pf.Signal, rir: np.ndarray) -> pf.Signal:
    audio = np.asarray(audio_pf.time).squeeze()
    if audio_pf.sampling_rate != RIR_SAMPLE_RATE:
        audio = resample_audio(audio, audio_pf.sampling_rate, RIR_SAMPLE_RATE)

    orig_rms = compute_rms(audio)
    localized = convolve_with_rir(audio, rir)

    new_rms = compute_rms(localized)
    if orig_rms > 0 and new_rms > 0:
        localized = localized * (orig_rms / new_rms)

    peak = float(np.max(np.abs(localized))) if localized.size else 0.0
    if peak > 0.999:
        localized = localized * (0.999 / peak)

    return pf.Signal(localized.T, sampling_rate=RIR_SAMPLE_RATE, fft_norm="none")


def main():
    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(f"Input audio not found: {INPUT_AUDIO}")

    for label, rir_path in BRIR_PATHS.items():
        if not rir_path.exists():
            raise FileNotFoundError(f"BRIR file not found for {label}: {rir_path}")

    print(f"Loading input audio: {INPUT_AUDIO}")
    audio_pf = load_mono_signal(INPUT_AUDIO)
    audio_pf = trim_input_duration(audio_pf, INPUT_DURATION_SEC)
    print(f"Using input duration: {audio_pf.n_samples / audio_pf.sampling_rate:.3f}s")

    input_stem = INPUT_AUDIO.stem

    for label, rir_path in BRIR_PATHS.items():
        print(f"\nProcessing target: {label}")
        rir = load_rir(rir_path)
        print(f"Loading BRIR: {rir_path}")
        print(f"BRIR shape: {rir.shape}, assumed sample rate: {RIR_SAMPLE_RATE} Hz")

        binaural = localise_with_rir(audio_pf, rir)
        output_path = OUTPUT_DIR / f"{input_stem}_{label}.wav"
        save_binaural(output_path, binaural, output_duration_sec=OUTPUT_DURATION_SEC)

    print("\nDone.")


if __name__ == "__main__":
    main()
