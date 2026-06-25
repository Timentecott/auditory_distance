# file as input, play the file whilst recording, using input and output files as specificed in defaults.npy
# save the output in recordings with the original file name plus _recorded

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from matplotlib import pyplot as plt

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4
INPUT_CHANNELS = [1, 2]
OUTPUT_CHANNELS = [1, 2]


def _parse_args():
    parser = argparse.ArgumentParser(description='Play an audio file or folder while recording using fixed audio settings.')
    parser.add_argument('audiofile', help='Path to the audio file or folder to play back while recording')
    return parser.parse_args()


def _read_wav_as_float(path):
    fs, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        max_value = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_value
    else:
        data = data.astype(np.float32)
    return fs, data


def _ensure_channel_layout(data, output_channels):
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if data.shape[1] == len(output_channels):
        return data

    if data.shape[1] == 1 and len(output_channels) > 1:
        return np.repeat(data, len(output_channels), axis=1)

    raise ValueError(
        f'Audio file has {data.shape[1]} channel(s), but output mapping expects {len(output_channels)} channel(s).'
    )


def _build_waveform_figure(data, fs, source_name):
    if data.ndim == 1:
        data = data[:, np.newaxis]

    waveform = np.mean(data, axis=1)
    time_axis = np.arange(waveform.shape[0]) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, waveform, linewidth=0.8)
    ax.set_title(f'Playing: {source_name}  |  Press space to skip')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _visualize_recording(recorded, fs, source_name):
    if recorded.ndim == 1:
        recorded = recorded[:, np.newaxis]

    time_axis = np.arange(recorded.shape[0]) / fs
    fig, axes = plt.subplots(recorded.shape[1], 1, sharex=True, figsize=(12, 3 * recorded.shape[1]))
    if recorded.shape[1] == 1:
        axes = [axes]

    fig.suptitle(f'Recorded RIR: {source_name}')
    for idx, ax in enumerate(axes):
        ax.plot(time_axis, recorded[:, idx], linewidth=0.8)
        ax.set_ylabel(f'Ch {idx + 1}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def _play_one_file(audio_path, output_path):
    fs, data = _read_wav_as_float(audio_path)
    data = _ensure_channel_layout(data, OUTPUT_CHANNELS)

    sd.default.device = [INPUT_DEVICE, OUTPUT_DEVICE]
    sd.default.samplerate = fs
    sd.default.dtype = 'float32'

    print('Input device:', INPUT_DEVICE)
    print('Output device:', OUTPUT_DEVICE)
    print('Input channels:', INPUT_CHANNELS, '(stereo)')
    print('Output channels:', OUTPUT_CHANNELS, '(stereo)')
    print('Playing:', audio_path)
    print('Press space in the waveform window to stop early.')

    stop_event = threading.Event()
    done_event = threading.Event()

    fig = _build_waveform_figure(data, fs, audio_path.name)

    def on_key(event):
        if event.key == ' ':
            stop_event.set()
            sd.stop()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)
    plt.pause(0.001)

    recorded = sd.playrec(data, samplerate=fs, input_mapping=INPUT_CHANNELS, output_mapping=OUTPUT_CHANNELS)
    start_time = time.monotonic()

    def _wait_for_audio():
        try:
            sd.wait()
        finally:
            done_event.set()

    wait_thread = threading.Thread(target=_wait_for_audio, daemon=True)
    wait_thread.start()

    while not done_event.is_set() and not stop_event.is_set():
        plt.pause(0.05)

    if stop_event.is_set() and not done_event.is_set():
        sd.stop()
        wait_thread.join()

    plt.close(fig)

    if stop_event.is_set():
        elapsed_frames = int((time.monotonic() - start_time) * fs)
        recorded = recorded[:max(1, min(elapsed_frames, recorded.shape[0]))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, fs, recorded.astype(np.float32))

    print('Saved recording to', output_path)
    _visualize_recording(recorded, fs, audio_path.name)


def _gather_audio_files(input_path):
    if input_path.is_file():
        return [input_path], input_path.parent

    if input_path.is_dir():
        files = sorted(input_path.rglob('*.wav'))
        return files, input_path

    raise FileNotFoundError(f'Input path not found: {input_path}')


def main():
    args = _parse_args()

    input_path = Path(args.audiofile)
    audio_files, source_root = _gather_audio_files(input_path)

    if not audio_files:
        raise FileNotFoundError(f'No WAV files found in {input_path}')

    recordings_root = Path('recordings')
    if input_path.is_file():
        for audio_path in audio_files:
            output_path = recordings_root / f'{audio_path.stem}_recorded.wav'
            _play_one_file(audio_path, output_path)
    else:
        output_folder = recordings_root / f'{input_path.name}_recorded'
        for audio_path in audio_files:
            relative_path = audio_path.relative_to(source_root)
            output_path = (output_folder / relative_path).with_name(f'{audio_path.stem}_recorded.wav')
            _play_one_file(audio_path, output_path)


if __name__ == '__main__':
    main()
