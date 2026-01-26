# -*- coding: utf-8 -*-
"""
BRIR Localization Script - Batch Processing
Processes all audio files with one BRIR SOFA file
"""

# import the required packages
import pyfar as pf
import sofar as sofa
import numpy as np
import soundfile as sf
import os
from pathlib import Path

# ============ CONFIGURATION ============
# SOFA file
SOFA_PATH = r"C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\BRIRs_from_a_room\B\020.sofa"

# Folder structure for batch processing
BASE_DIR = Path(r"C:\Users\tim_e\source\repos\auditory_distance")
INPUT_BASE = BASE_DIR / "original_audios"
OUTPUT_BASE = BASE_DIR / "localised_stimuli_B20"

# Audio file extensions to process
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff'}

# Spatialization parameters (used to find closest BRIR in SOFA file)
TARGET_AZIMUTH = 45  # degrees (left-front)
TARGET_ELEVATION = 0  # degrees
TARGET_DISTANCE = 1.0  # meters

print("=" * 70)
print("BRIR Audio Localization - BATCH PROCESSING")
print("=" * 70)
print(f"SOFA file: {SOFA_PATH}")
print(f"Input folder: {INPUT_BASE}")
print(f"Output folder: {OUTPUT_BASE}")
print(f"Target position: {TARGET_AZIMUTH} deg azimuth, {TARGET_ELEVATION} deg elevation, {TARGET_DISTANCE}m")
print("=" * 70)

# ============ GENERATE OUTPUT FILENAME ============
def generate_output_filename(input_audio_path, sofa_path):
    """
    Generate output filename from input audio and SOFA file paths
    
    Format: <input_stem>_BRIR_localised_<sofa_identifier>.wav
    Example: ISTS-V1.0_60s_24bit_1_BRIR_localised_BRIRs_from_a_room_B_020.wav
    """
    # Get input filename without extension
    input_stem = Path(input_audio_path).stem
    
    # Extract SOFA identifier from path
    sofa_path_obj = Path(sofa_path)
    
    # Get the last 3 parts of the path (e.g., BRIRs_from_a_room/B/020.sofa)
    parts = sofa_path_obj.parts
    
    # Take last 3 parts (or fewer if path is shorter)
    meaningful_parts = parts[-3:] if len(parts) >= 3 else parts
    
    # Remove .sofa extension from last part if present
    meaningful_parts = list(meaningful_parts)
    if meaningful_parts[-1].endswith('.sofa'):
        meaningful_parts[-1] = meaningful_parts[-1][:-5]
    
    # Join with underscores
    sofa_id = '_'.join(meaningful_parts)
    
    # Remove any remaining invalid filename characters
    invalid_chars = [':', '/', '\\', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        sofa_id = sofa_id.replace(char, '_')
    
    # Combine: <input>_BRIR_localised_<sofa_id>.wav
    output_filename = f"{input_stem}_BRIR_localised_{sofa_id}.wav"
    
    return output_filename

# ============ LOAD SOFA FILE ============
print("\nLoading SOFA file...")
try:
    sofa_data = sofa.read_sofa(SOFA_PATH, verify=False)
    print(f"? Successfully loaded: {Path(SOFA_PATH).name}")
except Exception as e:
    print(f"? Failed to load SOFA file!")
    print(f"Error: {e}")
    exit(1)

# Extract IRs and source positions from SOFA file
source_positions = sofa_data.SourcePosition

# SOFA format: [azimuth (degrees), elevation (degrees), radius (meters)]
azimuth_rad = np.deg2rad(source_positions[:, 0])
elevation_rad = np.deg2rad(source_positions[:, 1])
radius = source_positions[:, 2]

# Create Coordinates object from spherical data
sources = pf.Coordinates.from_spherical_elevation(
    azimuth_rad,
    elevation_rad,
    radius
)

print(f"Loaded {sources.csize} source positions")

# Find the target BRIR
target_position = pf.Coordinates.from_spherical_elevation(
    np.deg2rad(TARGET_AZIMUTH),
    np.deg2rad(TARGET_ELEVATION),
    TARGET_DISTANCE
)

index, distance = sources.find_nearest(target_position)
print(f"Using BRIR at index {index}, distance from target: {distance:.4f} m")
print(f"Measurement distance from SOFA: {radius[index[0]]:.2f} m")

# Extract the BRIR (load once, reuse for all files)
ir_data = sofa_data.Data_IR[index[0]]
sampling_rate_ir = sofa_data.Data_SamplingRate[0] if hasattr(sofa_data.Data_SamplingRate, '__len__') else sofa_data.Data_SamplingRate
ir_pf = pf.Signal(ir_data, sampling_rate_ir, fft_norm='none')
print(f"BRIR sampling rate: {sampling_rate_ir} Hz")
print(f"BRIR shape: {ir_pf.cshape}")

# ============ BATCH PROCESSING FUNCTIONS ============

def process_audio_file(input_path, output_path, ir_pf):
    """Process a single audio file for batch mode"""
    try:
        # Load audio file
        audio, fs_audio = sf.read(str(input_path))
        audio_pf = pf.Signal(audio.T, sampling_rate=fs_audio)
        
        # Convert to mono if stereo
        if audio_pf.cshape[0] > 1:
            audio_pf = pf.Signal(np.mean(audio_pf.time, axis=0, keepdims=True), sampling_rate=fs_audio)
        
        # Resample if needed
        if audio_pf.sampling_rate != ir_pf.sampling_rate:
            audio_pf = pf.dsp.resample(audio_pf, ir_pf.sampling_rate)
        
        # Convolve with BRIR
        binaural_signal = pf.dsp.convolve(audio_pf, ir_pf)
        
        # Save output
        sf.write(str(output_path), binaural_signal.time.T, int(binaural_signal.sampling_rate))
        
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def process_folder(folder_path, output_base, ir_pf):
    """
    Process all audio files in a folder (recursively)
    
    Args:
        folder_path: Input folder path
        output_base: Base output folder path
        ir_pf: Preloaded BRIR signal
    
    Returns:
        Number of successfully processed files
    """
    folder_path = Path(folder_path)
    output_base = Path(output_base)
    
    # Get relative path from INPUT_BASE to preserve folder structure
    try:
        relative_path = folder_path.relative_to(INPUT_BASE)
    except ValueError:
        relative_path = Path("")
    
    output_folder = output_base / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files in this folder (not recursive yet)
    audio_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
    
    if not audio_files:
        return 0
    
    folder_name = relative_path if relative_path != Path("") else "root"
    print(f"\n{'='*70}")
    print(f"Processing folder: {folder_name} ({len(audio_files)} files)")
    print(f"{'='*70}")
    
    success_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        # Generate output filename
        output_filename = generate_output_filename(str(audio_file), SOFA_PATH)
        output_path = output_folder / output_filename
        
        print(f"[{i}/{len(audio_files)}] {audio_file.name}")
        print(f"  -> {output_filename}")
        
        if process_audio_file(audio_file, output_path, ir_pf):
            success_count += 1
            print(f"  ? Success")
        else:
            print(f"  ? Failed")
    
    return success_count

def process_all_folders(input_base, output_base, ir_pf):
    """
    Process all folders and subfolders
    
    Args:
        input_base: Base input folder
        output_base: Base output folder
        ir_pf: Preloaded BRIR signal
    
    Returns:
        Total number of files processed
    """
    input_base = Path(input_base)
    output_base = Path(output_base)
    
    # Find all subdirectories (including the base)
    all_folders = [input_base] + [f for f in input_base.rglob('*') if f.is_dir()]
    
    # Filter to only folders that contain audio files
    folders_with_audio = []
    for folder in all_folders:
        audio_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        if audio_files:
            folders_with_audio.append(folder)
    
    print(f"\nFound {len(folders_with_audio)} folders with audio files")
    
    total_processed = 0
    for folder in folders_with_audio:
        count = process_folder(folder, output_base, ir_pf)
        total_processed += count
    
    return total_processed

# ============ RUN BATCH PROCESSING ============

print("\n" + "=" * 70)
print("STARTING BATCH PROCESSING")
print("=" * 70)

total_processed = process_all_folders(INPUT_BASE, OUTPUT_BASE, ir_pf)

# ============ SUMMARY ============

print("\n" + "=" * 70)
print("BATCH PROCESSING COMPLETE")
print("=" * 70)
print(f"Total files processed: {total_processed}")
print(f"Output folder: {OUTPUT_BASE}")
print(f"Naming convention: <original_name>_BRIR_localised_BRIRs_from_a_room_B_020.wav")
print("=" * 70)