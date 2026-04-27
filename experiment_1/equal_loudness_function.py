from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


FILTER_Q = 5.0


def load_headphone_response(eq_file: Path):
    freqs = []
    comp_left = []
    comp_right = []
    sharp_left = []
    sharp_right = []

    with Path(eq_file).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue

            if len(values) < 5:
                continue

            freqs.append(values[0])
            comp_left.append(values[1])
            comp_right.append(values[2])
            sharp_left.append(values[3])
            sharp_right.append(values[4])

    if not freqs:
        raise ValueError(f"No valid EQ rows found in: {eq_file}")

    freqs = np.asarray(freqs, dtype=np.float64)
    order = np.argsort(freqs)

    return {
        "freq_hz": freqs[order],
        "comp_left_db": np.asarray(comp_left, dtype=np.float64)[order],
        "comp_right_db": np.asarray(comp_right, dtype=np.float64)[order],
        "sharp_left_db": np.asarray(sharp_left, dtype=np.float64)[order],
        "sharp_right_db": np.asarray(sharp_right, dtype=np.float64)[order],
    }


def _design_peaking_sos(f0_hz: float, gain_db: float, q: float, sr: int):
    nyquist = sr / 2.0
    if f0_hz <= 0 or f0_hz >= nyquist:
        return None

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (f0_hz / sr)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    if a0 == 0:
        return None

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return np.array([b0, b1, b2, 1.0, a1, a2], dtype=np.float64)


def _build_filterbank_sos(eq_freq_hz: np.ndarray, eq_gain_db: np.ndarray, sr: int, q: float):
    sos_rows = []
    for f0, g in zip(eq_freq_hz, eq_gain_db):
        if abs(float(g)) < 1e-9:
            continue
        row = _design_peaking_sos(float(f0), float(g), q, sr)
        if row is not None:
            sos_rows.append(row)

    if not sos_rows:
        return None
    return np.vstack(sos_rows)


def _apply_eq_channel(audio_1d: np.ndarray, sr: int, eq_freq_hz: np.ndarray, eq_gain_db: np.ndarray, q: float) -> np.ndarray:
    sos = _build_filterbank_sos(eq_freq_hz, eq_gain_db, sr, q)
    if sos is None:
        return audio_1d.astype(np.float32)

    out = signal.sosfilt(sos, audio_1d.astype(np.float64))
    return out.astype(np.float32)


def apply_equal_loudness_to_audio(
    audio: np.ndarray,
    sr: int,
    eq_file: Path,
    use_sharpened_columns: bool = False,
    normalize_if_clipping: bool = True,
    q: float = FILTER_Q,
) -> np.ndarray:
    """Apply personalized EQ to an in-memory audio array and return the EQ-processed audio."""
    profile = load_headphone_response(Path(eq_file))

    if use_sharpened_columns:
        left_db = profile["sharp_left_db"]
        right_db = profile["sharp_right_db"]
    else:
        left_db = profile["comp_left_db"]
        right_db = profile["comp_right_db"]

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        mono_db = 0.5 * (left_db + right_db)
        processed = _apply_eq_channel(audio, sr, profile["freq_hz"], mono_db, q=q)
    else:
        channels = [_apply_eq_channel(audio[:, 0], sr, profile["freq_hz"], left_db, q=q)]

        if audio.shape[1] > 1:
            channels.append(_apply_eq_channel(audio[:, 1], sr, profile["freq_hz"], right_db, q=q))
            for c in range(2, audio.shape[1]):
                channels.append(audio[:, c].astype(np.float32))

        processed = np.column_stack(channels)

    if normalize_if_clipping:
        peak = float(np.max(np.abs(processed))) if processed.size else 0.0
        if peak > 1.0:
            processed = processed / peak

    return processed


def apply_equal_loudness_to_file(
    input_audio_file: Path,
    eq_file: Path,
    use_sharpened_columns: bool = False,
    normalize_if_clipping: bool = True,
    q: float = FILTER_Q,
):
    """Load an audio file, apply personalized EQ, and return (eq_audio, sample_rate)."""
    input_audio_file = Path(input_audio_file)
    eq_file = Path(eq_file)

    if not input_audio_file.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_file}")
    if not eq_file.exists():
        raise FileNotFoundError(f"EQ file not found: {eq_file}")

    audio, sr = sf.read(str(input_audio_file), always_2d=False)
    eq_audio = apply_equal_loudness_to_audio(
        audio=audio,
        sr=sr,
        eq_file=eq_file,
        use_sharpened_columns=use_sharpened_columns,
        normalize_if_clipping=normalize_if_clipping,
        q=q,
    )
    return eq_audio, sr

