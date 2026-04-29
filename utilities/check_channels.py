"""Scan audio stimuli and report channel counts and sample rates.

Usage:
    python check_channels.py

Outputs a table to stdout and writes a CSV 'channel_report.csv' in the same folder.
"""
import os
import glob
import soundfile as sf
import csv

base_dir = os.path.dirname(__file__)
folders = [
    os.path.join(base_dir, 'localised_stimuli'),
    os.path.join(base_dir, 'loudspeaker_stimuli')
]
_audio_exts = ['wav','flac','mp3','aiff','ogg']

def list_audio(folder):
    files = []
    for ext in _audio_exts:
        files.extend(glob.glob(os.path.join(folder, '**', f'*.{ext}'), recursive=True))
    return sorted(files)

report = []
for folder in folders:
    if not os.path.isdir(folder):
        continue
    files = list_audio(folder)
    for f in files:
        try:
            info = sf.info(f)
            channels = info.channels
            samplerate = info.samplerate
            frames = info.frames
            duration = frames / samplerate if samplerate else 0
            report.append({'file': os.path.relpath(f, base_dir),
                           'folder': os.path.relpath(folder, base_dir),
                           'channels': channels,
                           'samplerate': samplerate,
                           'frames': frames,
                           'duration_s': round(duration, 3),
                           'subtype': info.subtype})
        except Exception as e:
            report.append({'file': os.path.relpath(f, base_dir),
                           'folder': os.path.relpath(folder, base_dir),
                           'channels': 'ERROR',
                           'samplerate': 'ERROR',
                           'frames': 'ERROR',
                           'duration_s': 'ERROR',
                           'subtype': str(e)})

# Print summary
print(f"Found {len(report)} audio files")
channels_count = {}
for r in report:
    ch = r['channels']
    channels_count[ch] = channels_count.get(ch, 0) + 1

print('\nChannel distribution:')
for ch, cnt in sorted(channels_count.items()):
    print(f"  {ch}: {cnt}")

# Print details header
print('\nDetails: file, folder, channels, samplerate, duration_s, subtype')
for r in report:
    print(f"{r['file']}, {r['folder']}, {r['channels']}, {r['samplerate']}, {r['duration_s']}, {r['subtype']}")

# Write CSV
csv_path = os.path.join(base_dir, 'channel_report.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['file','folder','channels','samplerate','frames','duration_s','subtype']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in report:
        writer.writerow(r)

print(f"\nWrote report to {csv_path}")
