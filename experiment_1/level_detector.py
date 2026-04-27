"""Live microphone level detector."""

from __future__ import annotations

import queue
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48_000
BLOCK_DURATION = 0.1  # seconds
INPUT_DEVICE = None   # set to an input device index if needed
CHANNELS = 1


def dbfs(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-10))


def list_input_devices() -> None:
    print("\nAvailable input devices:")
    for idx, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(f"  {idx}: {device['name']} ({device['max_input_channels']} ch)")


def main() -> None:
    list_input_devices()
    print("\nPress Ctrl+C to stop.\n")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=INPUT_DEVICE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
            callback=callback,
        ):
            while True:
                data = audio_queue.get()
                peak = float(np.max(np.abs(data)))
                rms = float(np.sqrt(np.mean(np.square(data))))
                print(
                    f"Peak: {peak:0.6f} ({dbfs(peak):6.1f} dBFS)  "
                    f"RMS: {rms:0.6f} ({dbfs(rms):6.1f} dBFS)",
                    end="\r",
                    flush=True,
                )
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

