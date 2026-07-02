import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import fftconvolve
import os
from pathlib import Path

#this file takes one binaural room impulse response and convolves it with audio files to create localised audio files
RIR_path = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\in_situ_lab_headphones_4sec_5rep\RIR.npy"
input_folder = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\original_audios"
output_folder = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\in_situ_stimuli"

#read RIR
rir = np.load(RIR_path)
print(f"RIR shape: {rir.shape}")
print(f"RIR dtype: {rir.dtype}")

# NOTE: The RIR sample rate needs to match the audio sample rate
# RIR has 52920 samples and should be ~4 seconds
# Therefore: sample_rate = 52920 / 4 = 13230 Hz

rir_sr = 44100  # 52920 samples / 4 seconds
print(f"RIR sample rate set to: {rir_sr} Hz (calculated for 4 second duration)\n")

# Find all WAV files in the input folder and subfolders
wav_files = list(Path(input_folder).rglob("*.wav"))
print(f"Found {len(wav_files)} WAV files to process\n")

for wav_file in wav_files:
    try:
        # Get relative path from input folder
        relative_path = wav_file.relative_to(input_folder)
        
        # Create output path with same structure
        output_path = Path(output_folder) / relative_path
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {relative_path}")
        
        #read audio file 
        audio_data, audio_sr = librosa.load(str(wav_file), sr=None, mono=False)
        print(f"  Audio sample rate: {audio_sr} Hz, shape: {audio_data.shape}")
        
        #if needed, resample audio file to match RIR sample rate
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :]
        
        if audio_sr != rir_sr:
            print(f"  Resampling from {audio_sr} Hz to {rir_sr} Hz")
            audio_data = librosa.resample(audio_data, orig_sr=audio_sr, target_sr=rir_sr)
            audio_sr = rir_sr
        
        #if needed, convert audio file to mono
        if audio_data.ndim > 1 and audio_data.shape[0] > 1:
            audio_mono = np.mean(audio_data, axis=0)
        else:
            audio_mono = audio_data.flatten()
        
        #convolve audio file with left channel of RIR 
        left_convolved = fftconvolve(audio_mono, rir[:, 0], mode='full')
        
        #convolve audio file with right channel of RIR
        right_convolved = fftconvolve(audio_mono, rir[:, 1], mode='full')
        
        #combine left and right convolved audio into a stereo audio file
        stereo_audio = np.vstack([left_convolved, right_convolved])
        
        # Normalize audio to prevent clipping and ensure proper amplitude
        max_val = np.max(np.abs(stereo_audio))
        if max_val > 0:
            stereo_audio = stereo_audio / max_val * 0.95  # Scale to 95% of max to avoid clipping
        
       # Convert to int16 format
        stereo_audio_int16 = np.clip(stereo_audio * 32767, -32768, 32767).astype(np.int16)
        
        #save stereo audio file to disk
        wavfile.write(str(output_path), audio_sr, stereo_audio_int16.T)
        print(f"  Saved to: {output_path}\n")
        
    except Exception as e:
        print(f"  ERROR processing {wav_file}: {str(e)}\n")

print("Processing complete!")
