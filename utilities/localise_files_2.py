# -*- coding: utf-8 -*-
#following 
#pysoundfile python-sofa librosa 
import numpy as np
import matplotlib.pyplot as plt
import sys, glob 
import soundfile as sf
import sofa
import sofar
import librosa
from scipy import signal
#from IPython.display import Audio


source = r'C:\Users\tim_e\source\repos\auditory_distance\original_audios\environment\blackbird_bbc_01.wav'
sofa_file = r'C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\HRIR_CIRC360.sofa'
output = r'C:\Users\tim_e\source\repos\auditory_distance\localised_stimuli\environment\blackbird_bbc_01_localised_right.wav'



def findnearest(array, value):
    """Find nearest value in array and return value and index"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# Open SOFA file 
print("Loading SOFA file...")
HRTF = sofa.Database.open(sofa_file)

fs_H = HRTF.Data.SamplingRate.get_values()[0]
print(f"HRTF Sample rate: {fs_H} Hz")

# Get available positions in spherical coordinates
position = HRTF.Source.Position.get_values(system='spherical')
print(f"Available positions: {position.shape[0]} measurements")
print(f"Position format: [azimuth, elevation, distance]")

# Target position
angle = 45  # degrees (azimuth)
elevation = 0  # degrees
distance = 1.0  # metres

# Find nearest available HRTF measurement
print(f"\nTarget position: azimuth={angle}deg, elevation={elevation}deg, distance={distance}m")

# Find nearest azimuth
azimuths = position[:, 0]
elevations = position[:, 1]
distances = position[:, 2]

# Debug: Print available azimuths to understand the range
unique_azimuths = np.unique(azimuths)
print(f"\nAvailable azimuths: min={unique_azimuths.min()}, max={unique_azimuths.max()}")
print(f"First few azimuths: {unique_azimuths[:10]}")

# Convert target angle to match SOFA convention
# If SOFA uses 0-360 range, convert negative angles
target_azimuth = angle
if unique_azimuths.min() >= 0 and unique_azimuths.max() > 180:
    # SOFA file uses 0-360 convention
    if target_azimuth < 0:
        target_azimuth = 360 + target_azimuth  # -45 becomes 315
    print(f"SOFA uses 0-360 convention. Converting {angle}deg to {target_azimuth}deg")
else:
    print(f"SOFA uses -180 to +180 convention")

nearest_az, idx_az = findnearest(azimuths, target_azimuth)
nearest_el, idx_el = findnearest(elevations, elevation)

# Find best matching position (considering both azimuth and elevation)
# Calculate angular distance for each HRTF position
angular_distances = np.sqrt((azimuths - target_azimuth)**2 + (elevations - elevation)**2)
best_idx = np.argmin(angular_distances)

actual_az = azimuths[best_idx]
actual_el = elevations[best_idx]
actual_dist = distances[best_idx]

print(f"Nearest HRTF: azimuth={actual_az}deg, elevation={actual_el}deg, distance={actual_dist}m (index {best_idx})")

# Get HRIR (Head-Related Impulse Response) for left and right ears
IR = HRTF.Data.IR.get_values()
HRIR_L = IR[best_idx, 0, :]  # Left ear
HRIR_R = IR[best_idx, 1, :]  # Right ear

print(f"HRIR length: {len(HRIR_L)} samples")

# Diagnostic: Check if HRIRs are actually different for left and right
hrir_diff = np.sum(np.abs(HRIR_L - HRIR_R))
print(f"HRIR L/R difference: {hrir_diff:.6f} (should be > 0 for spatial effect)")

# Check HRIR peak values
print(f"HRIR_L peak: {np.max(np.abs(HRIR_L)):.6f}")
print(f"HRIR_R peak: {np.max(np.abs(HRIR_R)):.6f}")

# Check for common issues
if hrir_diff < 0.001:
    print("WARNING: Left and right HRIRs are nearly identical - no spatial effect!")
if np.max(np.abs(HRIR_L)) < 0.001:
    print("WARNING: HRIRs have very low amplitude - may not be loaded correctly!")

# Load source audio
print(f"\nLoading source audio: {source}")
audio, fs_audio = sf.read(source)
print(f"Audio sample rate: {fs_audio} Hz, Duration: {len(audio)/fs_audio:.2f}s")

# Convert stereo to mono if needed
if audio.ndim > 1:
    print("Converting stereo to mono...")
    audio = audio.mean(axis=1)

# Resample audio to match HRTF sample rate if needed
if fs_audio != fs_H:
    print(f"Resampling audio from {fs_audio} Hz to {fs_H} Hz...")
    audio = librosa.resample(audio, orig_sr=fs_audio, target_sr=fs_H)
    fs_audio = fs_H

# Convolve audio with HRIRs to create binaural output
print("\nConvolving audio with HRIRs...")
binaural_L = signal.fftconvolve(audio, HRIR_L, mode='full')
binaural_R = signal.fftconvolve(audio, HRIR_R, mode='full')

# Diagnostic: Check output levels and differences
print(f"Binaural output - L peak: {np.max(np.abs(binaural_L)):.6f}, R peak: {np.max(np.abs(binaural_R)):.6f}")
lr_ratio = np.max(np.abs(binaural_L)) / (np.max(np.abs(binaural_R)) + 1e-10)
print(f"L/R ratio: {lr_ratio:.3f} (for azimuth={actual_az}deg, expect L>R if left, R>L if right)")

# Check ITD (Interaural Time Difference) - a key spatial cue
# Find peaks in left and right channels
peak_L_idx = np.argmax(np.abs(binaural_L[:int(fs_H*0.1)]))  # First 100ms
peak_R_idx = np.argmax(np.abs(binaural_R[:int(fs_H*0.1)]))
itd_samples = peak_L_idx - peak_R_idx
itd_microsec = (itd_samples / fs_H) * 1000000
print(f"ITD (Interaural Time Difference): {itd_microsec:.1f} microseconds ({itd_samples} samples)")
if abs(itd_microsec) < 10:
    print("WARNING: ITD is very small - spatial effect may be weak!")

# Combine left and right channels
binaural_output = np.column_stack([binaural_L, binaural_R])

# Normalize to prevent clipping
max_val = np.max(np.abs(binaural_output))
if max_val > 0.99:
    print(f"Normalizing audio (peak: {max_val:.3f})")
    binaural_output = binaural_output * (0.99 / max_val)

# Save output
print(f"\nSaving localized audio to: {output}")
sf.write(output, binaural_output, int(fs_H))

print(f"\n? Localization complete!")
print(f"  Input: {source}")
print(f"  Output: {output}")
print(f"  Position: azimuth={actual_az}deg, elevation={actual_el}deg")
print(f"  Output duration: {len(binaural_output)/fs_H:.2f}s")

