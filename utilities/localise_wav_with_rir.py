     #!/usr/bin/env python3
     """Localise a WAV file by convolving it with a single RIR.

     Usage:
         python utilities/localise_wav_with_rir.py --input dry.wav --rir room_rir.wav --output localized.wav

     The input audio is mixed to mono before convolution. The RIR must be mono or stereo;
     mono RIRs are duplicated to both output channels.
     """

     from __future__ import annotations

     import argparse
     from pathlib import Path

     import numpy as np
     import soundfile as sf
     from scipy import signal


     def load_audio(path: Path) -> tuple[np.ndarray, int]:
         audio, sr = sf.read(str(path), always_2d=False)
         audio = np.asarray(audio, dtype=np.float32)
         return audio, int(sr)


     def to_mono(audio: np.ndarray) -> np.ndarray:
         if audio.ndim == 1:
             return audio
         return np.mean(audio, axis=1)


     def load_rir(path: Path) -> np.ndarray:
         rir, _ = load_audio(path)
         if rir.ndim == 1:
             rir = np.column_stack((rir, rir))
         elif rir.ndim == 2 and rir.shape[1] == 1:
             rir = np.column_stack((rir[:, 0], rir[:, 0]))
         elif rir.ndim == 2 and rir.shape[1] >= 2:
             rir = rir[:, :2]
         else:
             raise ValueError("RIR must be mono or stereo")
         return rir


     def convolve_source_with_rir(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
         left = signal.fftconvolve(source, rir[:, 0], mode="full")
         right = signal.fftconvolve(source, rir[:, 1], mode="full")
         return np.column_stack((left, right))


     def localise_wav(input_path: Path, rir_path: Path, output_path: Path) -> bool:
         try:
             audio_signal, sr_audio = load_audio(input_path)
             audio_signal = to_mono(audio_signal)

             rir = load_rir(rir_path)
             localized = convolve_source_with_rir(audio_signal, rir)

             max_val = float(np.max(np.abs(localized))) if localized.size else 0.0
             if max_val > 0.0:
                 localized = localized / max_val

             output_path.parent.mkdir(parents=True, exist_ok=True)
             sf.write(str(output_path), localized, sr_audio)
             print(f"Processed: {input_path} -> {output_path}")
             return True
         except Exception as e:
             print(f"Error processing {input_path}: {e}")
             return False


     def main() -> int:
         parser = argparse.ArgumentParser(description="Localise a WAV file using a single RIR.")
         parser.add_argument("--input", "-i", required=True, type=Path, help="Path to the dry WAV file")
         parser.add_argument("--rir", "-r", required=True, type=Path, help="Path to the RIR WAV file")
         parser.add_argument("--output", "-o", required=True, type=Path, help="Path to write the localised WAV file")
         args = parser.parse_args()

         if not args.input.exists():
             raise FileNotFoundError(f"Input WAV not found: {args.input}")
         if not args.rir.exists():
             raise FileNotFoundError(f"RIR file not found: {args.rir}")

         success = localise_wav(args.input, args.rir, args.output)
         return 0 if success else 1


     if __name__ == "__main__":
         raise SystemExit(main())
