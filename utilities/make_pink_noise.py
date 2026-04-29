import pyfar as pf
import soundfile as sf

samplerate = 48000
time = 5  # seconds

# Generate pink noise first
pink_noise = pf.signals.noise(
    n_samples=samplerate * time,
    spectrum='pink',
    rms=1,  # Start with rms=1, will adjust after filtering
    sampling_rate=samplerate,
    seed=42
)

# Convert pink noise to brown noise by applying low-pass filter
# Brown noise = pink noise with additional -3dB/octave slope
# Use a gentle low-pass filter to achieve this
brown_noise = pf.dsp.filter.butterworth(pink_noise, 2, 1000, 'lowpass')

# Normalize to desired RMS level
target_rms = 0.1
current_rms = float(pf.dsp.rms(brown_noise)[0])
brown_noise = brown_noise * (target_rms / current_rms)

# Save to file
output_path = "brown_noise_5s.wav"
sf.write(output_path, brown_noise.time.T, int(brown_noise.sampling_rate))
print(f"Brown noise saved to: {output_path}")
print(f"Duration: {time} seconds")
print(f"RMS value: {float(pf.dsp.rms(brown_noise)[0]):.4f}")
print(f"Peak amplitude: {float(brown_noise.time.max()):.4f}")

