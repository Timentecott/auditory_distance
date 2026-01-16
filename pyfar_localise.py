#%matplotlib inline
# import the required packages
import pyfar as pf
import sofar as sofa
import numpy as np
import warnings
import matplotlib.pyplot as plt
import soundfile as sf

# load HRIR dataset from SOFA file
sofa_path = r"C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\HRIR_FULL2DEG.sofa"
sofa = sofa.read_sofa(sofa_path)


# Extract HRIRs and source positions from SOFA file
# SOFA typically stores positions in spherical coordinates (azimuth, elevation, radius)
source_positions = sofa.SourcePosition  # numpy array


# SOFA format: [azimuth (degrees), elevation (degrees), radius (meters)]
azimuth_rad = np.deg2rad(source_positions[:, 0])
elevation_rad = np.deg2rad(source_positions[:, 1])
radius = source_positions[:, 2]

# Create Coordinates object from spherical data in radians
sources = pf.Coordinates.from_spherical_elevation(
    azimuth_rad,     # azimuth in radians
    elevation_rad,   # elevation in radians
    radius           # radius in meters
)


sources.show()
plt.show()

# Load an example audio file to be spatialized
audio_path = r"C:\Users\tim_e\source\repos\auditory_distance\original_audios\environment\blackbird_bbc_01.wav"
audio, fs_audio = sf.read(audio_path)
audio_pf = pf.Signal(audio.T, sampling_rate=fs_audio)  # Use 'sampling_rate' not 'fs'

# Ensure audio is mono for convolution
if audio_pf.cshape[0] > 1:
    audio_pf = pf.Signal(np.mean(audio_pf.time, axis=0, keepdims=True), sampling_rate=fs_audio)

# Define target position for spatialization
target_azimuth = 45  # degrees
target_elevation = 0  # degrees
target_radius = 1.0  # meters
target_position = pf.Coordinates.from_spherical_elevation(
    np.deg2rad(target_azimuth),
    np.deg2rad(target_elevation),
    target_radius
)

# Find the nearest source position in the dataset
index, distance = sources.find_nearest(target_position)
print(f"Nearest source index: {index}, Distance: {distance:.4f} m")



# Extract corresponding HRIR
hrir_data = sofa.Data_IR[index[0]]  # index is a tuple, extract the integer
sampling_rate_hrir = sofa.Data_SamplingRate[0] if hasattr(sofa.Data_SamplingRate, '__len__') else sofa.Data_SamplingRate
hrir_pf = pf.Signal(hrir_data, sampling_rate_hrir, fft_norm='none')

# Plot HRIR in time and frequency domain
ax = pf.plot.time_freq(hrir_pf, label=["Left ear", "Right ear"])
ax[0].legend()
ax[1].legend()
plt.show()

# Resample audio to match HRIR sampling rate if needed
if audio_pf.sampling_rate != hrir_pf.sampling_rate:
    audio_pf = pf.dsp.resample(audio_pf, hrir_pf.sampling_rate)

# Convolve audio with HRIR to create binaural signal
binaural_signal = pf.dsp.convolve(audio_pf, hrir_pf)

print(f"Binaural signal shape: {binaural_signal.cshape}")
print(f"Binaural signal has {binaural_signal.cshape[0]} channels")

# Save the spatialized audio
output_path = r"C:\Users\tim_e\source\repos\auditory_distance\localised_stimuli\environment\blackbird_bbc_01_localised_right.wav"
sf.write(output_path, binaural_signal.time.T, int(binaural_signal.sampling_rate))
print(f"Spatialized audio saved to: {output_path}")

# Plot both channels of the binaural signal
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Plot left ear (channel 0)
pf.plot.time(binaural_signal[0], ax=axes[0])
axes[0].set_title('Left Ear')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')

# Plot right ear (channel 1)
pf.plot.time(binaural_signal[1], ax=axes[1])
axes[1].set_title('Right Ear')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()