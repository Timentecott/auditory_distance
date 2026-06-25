import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import fftconvolve
import sounddevice as sd

ISTS_FILENAME = 'ISTS-V1.0_60s_24bit.wav'
DEFAULT_RIR_FS = 44100


def _parse_args():
    parser = argparse.ArgumentParser(description='Play a room impulse response file and convolve it with ISTS.')
    parser.add_argument('rirfile', help='Path to the RIR file (.npy or .wav)')
    parser.add_argument('--istsfile', default=ISTS_FILENAME, help='Path to the source WAV file')
    parser.add_argument('--rirfs', type=int, default=DEFAULT_RIR_FS, help='Sample rate to use when the RIR is loaded from .npy')
    parser.add_argument('--save', action='store_true', help='Save the localized convolved audio to recordings/')
    return parser.parse_args()


def _read_wav_as_float(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', WavFileWarning)
        fs, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        max_value = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_value
    else:
        data = data.astype(np.float32)
    return fs, data


def _read_rir(path, rirfs):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        rir = np.load(path, allow_pickle=True)
        return rirfs, rir.astype(np.float32)
    return _read_wav_as_float(path)


def _as_channels(data):
    if data.ndim == 1:
        return data[:, np.newaxis]
    return data


def _channel_description(data):
    num_channels = 1 if data.ndim == 1 else data.shape[1]
    if num_channels == 1:
        return 'mono'
    if num_channels == 2:
        return 'stereo'
    return f'{num_channels}-channel'


def _play_audio(data, fs):
    sd.play(data, fs)
    try:
        while True:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        sd.stop()
        print('Playback interrupted.')
    finally:
        sd.stop()


def _play_rir_until_done(rir, fs):
    while True:
        _play_audio(rir, fs)
        answer = input('Press Enter to play the RIR again, or type c then Enter to continue: ').strip().lower()
        if answer == 'c':
            break


def _convolve_localized(source, rir):
    source = _as_channels(source)
    rir = _as_channels(rir)

    if source.shape[1] == rir.shape[1]:
        convolved = [fftconvolve(source[:, idx], rir[:, idx], mode='full') for idx in range(source.shape[1])]
    elif source.shape[1] == 1 and rir.shape[1] == 2:
        convolved = [fftconvolve(source[:, 0], rir[:, idx], mode='full') for idx in range(2)]
    elif source.shape[1] == 2 and rir.shape[1] == 1:
        convolved = [fftconvolve(source[:, idx], rir[:, 0], mode='full') for idx in range(2)]
    elif source.shape[1] == 1 and rir.shape[1] == 1:
        convolved = [fftconvolve(source[:, 0], rir[:, 0], mode='full')]
    else:
        raise ValueError(
            f'Cannot localize source with {source.shape[1]} channel(s) and RIR with {rir.shape[1]} channel(s).'
        )

    max_len = max(len(ch) for ch in convolved)
    output = np.zeros((max_len, len(convolved)), dtype=np.float32)
    for idx, ch in enumerate(convolved):
        output[:len(ch), idx] = ch.astype(np.float32)

    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak

    if output.shape[1] == 1:
        return output[:, 0]
    return output


def _plot_multichannel_waveform(data, fs, title):
    data = _as_channels(data)
    time_axis = np.arange(data.shape[0]) / fs
    fig, axes = plt.subplots(data.shape[1], 1, sharex=True, figsize=(12, 3 * data.shape[1]))
    if data.shape[1] == 1:
        axes = [axes]

    fig.suptitle(title)
    for idx, ax in enumerate(axes):
        ax.plot(time_axis, data[:, idx], linewidth=0.8)
        ax.set_ylabel(f'Ch {idx + 1}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    return fig


def _show_waveform(data, fs, title):
    fig = _plot_multichannel_waveform(data, fs, title)
    plt.show()


def main():
    args = _parse_args()

    rir_path = Path(args.rirfile)
    ists_path = Path(args.istsfile)

    rir_fs, rir = _read_rir(rir_path, args.rirfs)
    print(f'RIR: {_channel_description(rir)}')
    _show_waveform(rir, rir_fs, f'RIR waveform: {rir_path.name}')
    _play_rir_until_done(rir, rir_fs)

    source_fs, source = _read_wav_as_float(ists_path)
    print(f'Source: {_channel_description(source)}')

    if rir_fs != source_fs:
        raise ValueError(f'Sample rate mismatch: RIR is {rir_fs} Hz but source is {source_fs} Hz')

    convolved = _convolve_localized(source, rir)
    _show_waveform(convolved, source_fs, f'Localized convolved audio: {ists_path.name} × {rir_path.name}')
    _play_audio(convolved, source_fs)

    if args.save:
        recordings_dir = Path('recordings')
        recordings_dir.mkdir(exist_ok=True)
        output_name = f'{ists_path.stem}_localized_{rir_path.stem}.wav'
        output_path = recordings_dir / output_name
        wavfile.write(output_path, source_fs, convolved.astype(np.float32))
        print('Saved localized audio to', output_path)


if __name__ == '__main__':
    main()