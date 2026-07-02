#!/usr/bin/env python3
"""
Convert a .npy file containing audio samples to a .wav file.

Usage:
  python npy_to_wav.py input.npy output.wav --sr 44100 --bitdepth 16 --normalize

Assumptions:
- .npy contains a 1-D (n_samples,) or 2-D (n_samples, n_channels) numpy array.
- Float arrays are expected in range [-1.0, 1.0] (if not, use --normalize).
- Only 16-bit PCM output is supported.
"""

import argparse
import sys
import numpy as np
import wave


def load_array(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load .npy file: {e}")


def prepare_array(arr, bitdepth=16, normalize=False):
    if bitdepth != 16:
        raise ValueError("Only 16-bit output is supported in this script")

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        # Heuristic: if first dim is small (<=8) and less than second dim, it may be (channels, samples)
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
    else:
        raise ValueError("Only 1-D or 2-D arrays are supported")

    # Convert to float for normalization/scaling logic
    if np.issubdtype(arr.dtype, np.floating):
        data = arr.astype(np.float32)
        if normalize:
            peak = np.max(np.abs(data))
            if peak > 0:
                data = data / peak
        # Clip to [-1,1]
        data = np.clip(data, -1.0, 1.0)
        max_int16 = 2 ** 15 - 1
        int_data = (data * max_int16).astype(np.int16)
    elif np.issubdtype(arr.dtype, np.integer):
        # Convert integer types to int16, with clipping
        # Scale down if larger than int16 range
        info = np.iinfo(arr.dtype)
        data = arr.astype(np.int64)
        # If already within int16 range, just cast
        if info.min >= -32768 and info.max <= 32767:
            int_data = data.astype(np.int16)
        else:
            # scale to int16 range
            data_f = (data - info.min) / (info.max - info.min)  # 0..1
            data_f = data_f * 2.0 - 1.0  # -1..1
            max_int16 = 2 ** 15 - 1
            int_data = (data_f * max_int16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported array dtype: {arr.dtype}")

    return int_data


def write_wav(path, int_data, sr):
    # int_data: shape (n_samples, n_channels)
    n_channels = int_data.shape[1]
    n_frames = int_data.shape[0]
    sampwidth = 2  # bytes for 16-bit

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(int_data.flatten().tobytes())


def main():
    p = argparse.ArgumentParser(description='Convert .npy audio array to .wav (16-bit PCM)')
    p.add_argument('input', help='input .npy file')
    p.add_argument('output', help='output .wav file')
    p.add_argument('--sr', type=int, default=44100, help='sample rate (default: 44100)')
    p.add_argument('--bitdepth', type=int, default=16, help='output bit depth (only 16 supported)')
    p.add_argument('--normalize', action='store_true', help='normalize floating audio to peak=1 before scaling')
    args = p.parse_args()

    try:
        arr = load_array(args.input)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    try:
        int_data = prepare_array(arr, bitdepth=args.bitdepth, normalize=args.normalize)
    except Exception as e:
        print(f"Error preparing array: {e}", file=sys.stderr)
        sys.exit(3)

    # Ensure shape is (n_samples, n_channels)
    if int_data.ndim == 1:
        int_data = int_data.reshape(-1, 1)

    try:
        write_wav(args.output, int_data, args.sr)
    except Exception as e:
        print(f"Failed writing wav: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"Wrote {args.output}: {int_data.shape[0]} frames, {int_data.shape[1]} channels, {args.sr} Hz")


if __name__ == '__main__':
    main()
