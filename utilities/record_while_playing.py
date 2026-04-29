# Play multiple WAV files while simultaneously recording from in-ear microphone
# Records all files for LEFT ear first, then all files for RIGHT ear, then combines into stereo files
# Processes folders with subfolders and recreates folder structure in output

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS
# ============================================================================

# Input folder containing subfolders with WAV files
INPUT_FOLDER = r'C:\Users\tim_e\source\repos\auditory_distance\loudspeaker_stimuli'

# Output folder for binaural recordings (will recreate subfolder structure)
OUTPUT_FOLDER = r'C:\Users\tim_e\source\repos\auditory_distance\binaural_recordings'

# Optional: Save individual mono recordings
SAVE_INDIVIDUAL_EARS = False
INDIVIDUAL_EARS_FOLDER = r'C:\Users\tim_e\source\repos\auditory_distance\individual_ear_recordings'

# Audio device indices (use check_input_index.py to find correct indices)
OUTPUT_DEVICE = 5   # Speaker/headphone device index for playback
INPUT_DEVICE = 3    # In-ear microphone device index for recording

# Recording settings
ADD_PADDING = 0.5   # Extra seconds to record after playback ends
PRE_PADDING = 0.1   # Seconds to start recording before playback
PAUSE_BETWEEN_FILES = 0.5  # Seconds to pause between different files

# ============================================================================

def find_wav_files(input_folder):
    """
    Recursively find all WAV files in folder and subfolders.
    
    Args:
        input_folder: Root folder to search
        
    Returns:
        List of tuples: (absolute_path, relative_path)
    """
    input_path = Path(input_folder)
    wav_files = []
    
    # Find all .wav files recursively
    for wav_file in input_path.rglob('*.wav'):
        # Get relative path from input folder
        relative_path = wav_file.relative_to(input_path)
        wav_files.append((str(wav_file), relative_path))
    
    return wav_files

def play_and_record_single_ear(input_file, ear_name, file_number, total_files, 
                                output_device=None, input_device=None, 
                                pre_padding=0.1, post_padding=0.5):
    """
    Play an audio file while recording from one ear.
    
    Args:
        input_file: Path to input WAV file to play
        ear_name: "LEFT" or "RIGHT" for display
        file_number: Current file number (1-indexed)
        total_files: Total number of files to process
        output_device: Device index for playback
        input_device: Device index for recording
        pre_padding: Seconds to start recording before playback
        post_padding: Seconds to continue recording after playback ends
        
    Returns:
        recorded_audio: The recorded audio array (mono)
        sample_rate: Sample rate used
    """
    print("\n" + "="*70)
    print(f"RECORDING {ear_name} EAR - File {file_number}/{total_files}")
    print("="*70)
    
    # Load input audio
    print(f"\nLoading input audio: {Path(input_file).name}")
    audio_data, fs = sf.read(input_file)
    
    # Get info
    duration = len(audio_data) / fs
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Convert to mono for playback if stereo
    if audio_data.ndim > 1:
        audio_playback = audio_data.mean(axis=1)
    else:
        audio_playback = audio_data
    
    # Calculate recording duration
    record_duration = duration + pre_padding + post_padding
    record_samples = int(record_duration * fs)
    
    print(f"\nPlayback device: {output_device}")
    print(f"Recording device: {input_device}")
    print(f"Recording duration: {record_duration:.2f} seconds")
    
    # Prepare recording buffer
    recording = np.zeros((record_samples, 1), dtype=np.float32)
    
    print("\n" + "-"*70)
    if file_number == 1:
        print(f"Position the microphone in your {ear_name} ear.")
        print("Press ENTER when ready to start recording...")
        input()  # Wait for user
        
        # Countdown only for first file
        print("\nCountdown...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
    else:
        print(f"Keep microphone in {ear_name} ear.")
        print(f"Starting next recording immediately...")
    
    # Set up devices
    sd.default.device = (input_device, output_device)
    
    print(f"\nRED_CIRCLE RECORDING {ear_name} EAR")
    
    # Start recording
    rec_stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=fs,
        dtype='float32'
    )
    rec_stream.start()
    
    # Wait for pre-padding
    if pre_padding > 0:
        time.sleep(pre_padding)
    
    # Start playback
    print("SPEAKER PLAYBACK STARTED")
    sd.play(audio_playback, samplerate=fs, device=output_device)
    
    # Record while playing (and for post-padding)
    samples_read = 0
    while samples_read < record_samples:
        # Read available samples
        available = rec_stream.read_available
        if available > 0:
            chunk_size = min(available, record_samples - samples_read)
            chunk, _ = rec_stream.read(chunk_size)
            recording[samples_read:samples_read + len(chunk)] = chunk
            samples_read += len(chunk)
        else:
            time.sleep(0.01)  # Small delay if no data available
    
    # Stop everything
    sd.stop()
    rec_stream.stop()
    rec_stream.close()
    
    print(f"CHECK {ear_name} EAR RECORDING COMPLETE")
    print("-"*70)
    
    # Analysis
    max_val = np.max(np.abs(recording))
    rms = np.sqrt(np.mean(recording**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = 20 * np.log10(max_val + 1e-10)
    
    print(f"\n{ear_name} Ear Recording Analysis:")
    print(f"  Peak level: {max_val:.6f} ({peak_db:.1f} dBFS)")
    print(f"  RMS level:  {rms:.6f} ({rms_db:.1f} dBFS)")
    
    if max_val < 0.001:
        print(f"\n  WARNING WARNING: {ear_name} ear signal is very quiet!")
        print("    - Check microphone is in ear and not muted")
        print("    - Check microphone gain/volume")
    elif max_val > 0.99:
        print(f"\n  WARNING WARNING: {ear_name} ear signal is clipping!")
    else:
        print(f"\n  CHECK {ear_name} ear signal level looks good!")
    
    return recording.flatten(), fs

def combine_to_stereo(left_audio, right_audio, output_file, sample_rate):
    """
    Combine left and right ear recordings into a stereo file.
    
    Args:
        left_audio: Left ear recording (mono array)
        right_audio: Right ear recording (mono array)
        output_file: Path to save stereo WAV file
        sample_rate: Sample rate in Hz
    """
    # Make sure both recordings are the same length
    min_length = min(len(left_audio), len(right_audio))
    if len(left_audio) != len(right_audio):
        print(f"\nWARNING Length mismatch: Left={len(left_audio)}, Right={len(right_audio)}")
        print(f"  Trimming both to {min_length} samples")
        left_audio = left_audio[:min_length]
        right_audio = right_audio[:min_length]
    
    # Create stereo array: [left, right] in columns
    stereo_audio = np.column_stack([left_audio, right_audio])
    
    # Normalize if clipping
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 0.99:
        print(f"  Normalizing stereo (peak: {max_val:.3f})")
        stereo_audio = stereo_audio * (0.99 / max_val)
    
    # Save stereo file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path to string for soundfile
    sf.write(str(output_file), stereo_audio, sample_rate)
    print(f"  CHECK Saved: {output_path.name}")
    v

    return stereo_audio

def main():
    """Main function"""
    # Check if input folder exists
    input_folder = Path(INPUT_FOLDER)
    if not input_folder.exists():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        return
    
    if not input_folder.is_dir():
        print(f"ERROR: Input path is not a folder: {INPUT_FOLDER}")
        return
    
    # Find all WAV files in folder and subfolders
    print("\n" + "="*70)
    print("SCANNING INPUT FOLDER FOR WAV FILES")
    print("="*70)
    print(f"\nInput folder: {INPUT_FOLDER}")
    
    wav_files = find_wav_files(INPUT_FOLDER)
    
    if not wav_files:
        print("\nERROR: No WAV files found in input folder!")
        return
    
    total_files = len(wav_files)
    
    print("\n" + "="*70)
    print("BATCH BINAURAL RECORDING WITH IN-EAR MICROPHONE")
    print("="*70)
    print(f"\nFound {total_files} WAV files:")
    for i, (abs_path, rel_path) in enumerate(wav_files, 1):
        print(f"  {i}. {rel_path}")
    
    print("\nThis script will:")
    print(f"  1. Record ALL {total_files} files with microphone in LEFT ear")
    print(f"  2. Record ALL {total_files} files with microphone in RIGHT ear")
    print("  3. Combine each pair into stereo binaural recordings")
    print(f"  4. Recreate folder structure in output folder")
    print("\nMake sure you have:")
    print("  - In-ear microphone ready")
    print("  - Audio playback set to speakers/headphones")
    print("  - Time to complete all recordings (keep microphone positioned)")
    print("="*70)
    
    # Storage for recordings
    left_recordings = []
    right_recordings = []
    sample_rates = []
    
    # ========================================================================
    # STEP 1: Record all files with LEFT ear
    # ========================================================================
    print("\n\n" + "#"*70)
    print("### STEP 1: RECORDING LEFT EAR FOR ALL FILES ###")
    print("#"*70)
    
    for i, (input_file, relative_path) in enumerate(wav_files, 1):
        print(f"\nFile: {relative_path}")
        
        left_recording, fs = play_and_record_single_ear(
            input_file=input_file,
            ear_name="LEFT",
            file_number=i,
            total_files=total_files,
            output_device=OUTPUT_DEVICE,
            input_device=INPUT_DEVICE,
            pre_padding=PRE_PADDING,
            post_padding=ADD_PADDING
        )
        
        left_recordings.append(left_recording)
        sample_rates.append(fs)
        
        # Optional: Save individual left ear recording
        if SAVE_INDIVIDUAL_EARS:
            individual_folder = Path(INDIVIDUAL_EARS_FOLDER) / "left" / relative_path.parent
            individual_folder.mkdir(parents=True, exist_ok=True)
            
            output_path = individual_folder / f"{relative_path.stem}_left.wav"
            sf.write(str(output_path), left_recording, fs)
        
        # Pause between files (except after last file)
        if i < total_files:
            print(f"\nPausing {PAUSE_BETWEEN_FILES} seconds before next file...")
            time.sleep(PAUSE_BETWEEN_FILES)
    
    print("\n" + "="*70)
    print("LEFT EAR RECORDINGS COMPLETE!")
    print("="*70)
    
    # ========================================================================
    # STEP 2: Record all files with RIGHT ear
    # ========================================================================
    print("\n\n" + "#"*70)
    print("### STEP 2: RECORDING RIGHT EAR FOR ALL FILES ###")
    print("#"*70)
    print("\nNow move the microphone to your RIGHT ear.")
    print("Press ENTER when ready to start recording the right ear...")
    input()
    
    for i, (input_file, relative_path) in enumerate(wav_files, 1):
        print(f"\nFile: {relative_path}")
        
        right_recording, fs = play_and_record_single_ear(
            input_file=input_file,
            ear_name="RIGHT",
            file_number=i,
            total_files=total_files,
            output_device=OUTPUT_DEVICE,
            input_device=INPUT_DEVICE,
            pre_padding=PRE_PADDING,
            post_padding=ADD_PADDING
        )
        
        right_recordings.append(right_recording)
        
        # Optional: Save individual right ear recording
        if SAVE_INDIVIDUAL_EARS:
            individual_folder = Path(INDIVIDUAL_EARS_FOLDER) / "right" / relative_path.parent
            individual_folder.mkdir(parents=True, exist_ok=True)
            
            output_path = individual_folder / f"{relative_path.stem}_right.wav"
            sf.write(str(output_path), right_recording, fs)
        
        # Pause between files (except after last file)
        if i < total_files:
            print(f"\nPausing {PAUSE_BETWEEN_FILES} seconds before next file...")
            time.sleep(PAUSE_BETWEEN_FILES)
    
    print("\n" + "="*70)
    print("RIGHT EAR RECORDINGS COMPLETE!")
    print("="*70)
    
    # ========================================================================
    # STEP 3: Combine into stereo files
    # ========================================================================
    print("\n\n" + "#"*70)
    print("### STEP 3: COMBINING TO STEREO FILES ###")
    print("#"*70)
    
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for i, (input_file, relative_path) in enumerate(wav_files):
        # Create output path with same subfolder structure
        output_subfolder = output_folder / relative_path.parent
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Create output filename with _binaural_recording extension
        output_file = output_subfolder / f"{relative_path.stem}_binaural_recording.wav"
        
        print(f"\nProcessing {i+1}/{total_files}: {relative_path}")
        
        combine_to_stereo(
            left_recordings[i],
            right_recordings[i],
            output_file,
            sample_rates[i]
        )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "="*70)
    print("PARTY POPPER BATCH BINAURAL RECORDING COMPLETE!")
    print("="*70)
    print(f"\nProcessed {total_files} files")
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    print("\nGenerated files (with recreated folder structure):")
    for i, (input_file, relative_path) in enumerate(wav_files, 1):
        output_rel = relative_path.parent / f"{relative_path.stem}_binaural_recording.wav"
        print(f"  {i}. {output_rel}")
    
    if SAVE_INDIVIDUAL_EARS:
        print(f"\nIndividual ear recordings saved to: {INDIVIDUAL_EARS_FOLDER}")
    
    print("\nYou can now use these binaural recordings for spatial audio playback!")
    print("="*70)

if __name__ == "__main__":
    main()

