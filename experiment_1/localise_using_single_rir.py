import numpy as np
from pathlib import Path

import soundfile as sf
from scipy import signal


base_dir = Path(__file__).resolve().parent
input_root = base_dir / "original_audios"
output_root = base_dir / "in_situ_stimuli"
rir = base_dir / "RIR.npy"


def find_wav_files(root: Path):
    return sorted(
        candidate for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == ".wav"
    )


# load the rir
rir_data = np.load(rir)
if rir_data.ndim == 1:
    rir_data = np.column_stack([rir_data, rir_data])
elif rir_data.ndim == 2 and rir_data.shape[0] == 2 and rir_data.shape[1] > 2:
    rir_data = rir_data.T

if not input_root.exists():
    raise FileNotFoundError(f"Input folder not found: {input_root}")

wav_files = find_wav_files(input_root)
if not wav_files:
    raise FileNotFoundError(f"No WAV files found in: {input_root}")

for audio in wav_files:
    relative_path = audio.relative_to(input_root)
    output = output_root / relative_path

    # load the audio
    audio_data, fs = sf.read(str(audio))
    if audio_data.ndim == 1:
        audio_data = np.column_stack([audio_data, audio_data])
    elif audio_data.ndim == 2 and audio_data.shape[1] > 2:
        audio_data = audio_data[:, :2]

    # convolve l and r of audio with the stereo rir
    left = signal.fftconvolve(audio_data[:, 0], rir_data[:, 0], mode="full")
    right = signal.fftconvolve(audio_data[:, 1], rir_data[:, 1], mode="full")
    out = np.column_stack([left, right])

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out * (0.99 / peak)

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), out, fs)
    print(f"Saved binaural output to {output}")

