# -*- coding: utf-8 -*-
"""
Batch BRIR Localization Script
Processes all audio files in original_audios folder structure
Saves to localised_stimuli with suffix _localised_brir_1m_sadie
"""

# import the required packages
import pyfar as pf
import sofar as sofa
import numpy as np
import warnings
import matplotlib.pyplot as plt
import soundfile as sf
import os
from pathlib import Path

# ============ CONFIGURATION ============
USE_BRIR = True
SOFA_PATH = r"C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\SADIE_II_D1_BRIR_SOFA\D1_48K_24bit_0.3s_FIR_SOFA.sofa"
IR_TYPE = "BRIR"

# Spatialization parameters
TARGET_AZIMUTH = 45  # degrees (left-front)
TARGET_ELEVATION = 0  # degrees
TARGET_DISTANCE = 1.0  # meters

# Folder structure
BASE_DIR = Path(__file__).resolve().parent
INPUT_BASE = BASE_DIR / "original_audios"
OUTPUT_BASE = BASE_DIR / "localised_stimuli"
STIMULUS_TYPES = ['environment', 'ISTS', 'noise']

# Audio file extensions to process
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff'}

print("=" * 70)
print("BRIR Audio Localization - Batch Processing")
print("=" * 70)
print(f"SOFA file: {SOFA_PATH}")
print(f"Target position: {TARGET_AZIMUTH} deg azimuth, {TARGET_ELEVATION} deg elevation, {TARGET_DISTANCE}m")
print(f"Input folder: {INPUT_BASE}")
print(f"Output folder: {OUTPUT_BASE}")
print("=" * 70)

# ============ LOAD SOFA FILE ============
print("\nLoading SOFA file...")
sofa_data = sofa.read_sofa(SOFA_PATH)

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

# Extract the BRIR once (reuse for all files)
ir_data = sofa_data.Data_IR[index[0]]
sampling_rate_ir = sofa_data.Data_SamplingRate[0] if hasattr(sofa_data.Data_SamplingRate, '__len__') else sofa_data.Data_SamplingRate
ir_pf = pf.Signal(ir_data, sampling_rate_ir, fft_norm='none')
print(f"BRIR sampling rate: {sampling_rate_ir} Hz")
print(f"BRIR shape: {ir_pf.cshape}")

# ============ PROCESS FILES ============

def process_audio_file(input_path, output_path, ir_pf):
    """
    Process a single audio file with BRIR convolution
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save output file
        ir_pf: Preloaded BRIR as pyfar Signal
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio file
        audio, fs_audio = sf.read(str(input_path))
        audio_pf = pf.Signal(audio.T, sampling_rate=fs_audio)
        
        # Ensure audio is mono for convolution
        if audio_pf.cshape[0] > 1:
            audio_pf = pf.Signal(np.mean(audio_pf.time, axis=0, keepdims=True), sampling_rate=fs_audio)
        
        # Resample audio to match IR sampling rate if needed
        if audio_pf.sampling_rate != ir_pf.sampling_rate:
            audio_pf = pf.dsp.resample(audio_pf, ir_pf.sampling_rate)
        
        # Convolve audio with BRIR to create binaural signal
        binaural_signal = pf.dsp.convolve(audio_pf, ir_pf)
        
        # Save the spatialized audio
        sf.write(str(output_path), binaural_signal.time.T, int(binaural_signal.sampling_rate))
        
        return True
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def process_folder(stimulus_type):
    """
    Process all audio files in a stimulus type folder
    
    Args:
        stimulus_type: 'environment', 'ISTS', or 'noise'
    """
    input_folder = INPUT_BASE / stimulus_type
    output_folder = OUTPUT_BASE / stimulus_type
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = [f for f in input_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
    
    if not audio_files:
        print(f"\nWarning: No audio files found in {input_folder}")
        return 0
    
    print(f"\n{'='*70}")
    print(f"Processing {stimulus_type} folder: {len(audio_files)} files")
    print(f"{'='*70}")
    
    success_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        # Create output filename with suffix
        stem = audio_file.stem
        output_filename = f"{stem}_localised_brir_1m_sadie.wav"
        output_path = output_folder / output_filename
        
        print(f"[{i}/{len(audio_files)}] {audio_file.name}")
        print(f"  -> {output_filename}")
        
        # Process the file
        if process_audio_file(audio_file, output_path, ir_pf):
            success_count += 1
            print(f"  Success")
        else:
            print(f"  Failed")
    
    print(f"\n{stimulus_type}: {success_count}/{len(audio_files)} files processed successfully")
    return success_count


# ============ MAIN PROCESSING LOOP ============

total_processed = 0
total_files = 0

for stimulus_type in STIMULUS_TYPES:
    count = process_folder(stimulus_type)
    total_processed += count
    # Count total files
    input_folder = INPUT_BASE / stimulus_type
    if input_folder.exists():
        audio_files = [f for f in input_folder.iterdir() 
                       if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        total_files += len(audio_files)

# ============ SUMMARY ============

print("\n" + "=" * 70)
print("BATCH PROCESSING COMPLETE")
print("=" * 70)
print(f"Total files processed: {total_processed}/{total_files}")
print(f"Output folder: {OUTPUT_BASE}")
print(f"Naming convention: <original_name>_localised_brir_1m_sadie.wav")
print("=" * 70)