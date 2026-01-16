#this file converts audio files to mono and places them in the left channel of a stereo file,
# with the right channel silent.

import os
import glob
import soundfile as sf
import numpy as np

# Source folder with 3 subfolders containing audio files
source_dir = r'C:\Users\tim_e\source\repos\auditory_distance\original_audios'

# Destination folder for loudspeaker audio files
dest_dir = r'C:\Users\tim_e\source\repos\auditory_distance\loudspeaker_stimuli'

# Supported audio extensions
audio_exts = ('*.wav', '*.flac', '*.mp3', '*.aiff', '*.ogg')

def convert_to_loudspeaker_audio(input_path, output_path):
    """
    Convert audio to loudspeaker format:
    - Mix to mono if stereo/multi-channel
    - Create stereo output with audio only in left channel, right channel silent
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save output audio file
    """
    try:
        # Read audio file
        audio_data, sample_rate = sf.read(input_path)
        
        # Convert to mono if multi-channel
        if audio_data.ndim > 1:
            # Mix all channels to mono
            mono_audio = audio_data.mean(axis=1)
        else:
            # Already mono
            mono_audio = audio_data
        
        # Create stereo output: left channel = mono audio, right channel = silence
        loudspeaker_audio = np.column_stack([mono_audio, np.zeros_like(mono_audio)])
        
        # Write output file
        sf.write(output_path, loudspeaker_audio, sample_rate)
        print(f"  ? Converted: {os.path.basename(input_path)}")
        return True
    
    except Exception as e:
        print(f"  ? Error converting {os.path.basename(input_path)}: {e}")
        return False

def process_folder_structure():
    """
    Recreate folder structure from source_dir to dest_dir with converted audio files.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all subdirectories in source
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not subdirs:
        print(f"Warning: No subdirectories found in {source_dir}")
        return
    
    print(f"\nProcessing folders from: {source_dir}")
    print(f"Output directory: {dest_dir}\n")
    
    total_files = 0
    converted_files = 0
    
    # Process each subdirectory
    for subdir in subdirs:
        source_subdir = os.path.join(source_dir, subdir)
        dest_subdir = os.path.join(dest_dir, subdir)
        
        # Create destination subdirectory
        os.makedirs(dest_subdir, exist_ok=True)
        
        # Find all audio files in this subdirectory (recursively)
        audio_files = []
        for ext in audio_exts:
            audio_files.extend(glob.glob(os.path.join(source_subdir, '**', ext), recursive=True))
        
        if not audio_files:
            print(f"Folder '{subdir}': No audio files found")
            continue
        
        print(f"Folder '{subdir}': Found {len(audio_files)} audio files")
        
        # Process each audio file
        for audio_file in audio_files:
            total_files += 1
            
            # Get relative path from source subdirectory
            rel_path = os.path.relpath(audio_file, source_subdir)
            
            # Create output path maintaining subdirectory structure
            output_file = os.path.join(dest_subdir, rel_path)
            
            # Create output subdirectories if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert audio file
            if convert_to_loudspeaker_audio(audio_file, output_file):
                converted_files += 1
        
        print()
    
    print(f"Conversion complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully converted: {converted_files}")
    print(f"Failed: {total_files - converted_files}")

if __name__ == "__main__":
    process_folder_structure()