import sounddevice as sd
import numpy as np

print("Available audio output devices:")
devices = sd.query_devices()
for idx, device in enumerate(devices):
    if device['max_output_channels'] > 0:
        print(f"Index {idx}: {device['name']}")

# Generate a test tone (440 Hz sine wave, 1 second)
duration = 1.0  # seconds
fs = 48000  # sample rate
t = np.linspace(0, duration, int(fs * duration))
frequency = 440  # Hz (A4 note)
test_tone = 0.3 * np.sin(2 * np.pi * frequency * t)

while True:
    # Ask user to input index of desired output device
    user_input = input("\nEnter device index to test (or 'q' to quit): ")
    
    if user_input.lower() == 'q':
        print("Exiting...")
        break
    
    try:
        device_index = int(user_input)
        print(f"\nPlaying test tone through device {device_index}: {devices[device_index]['name']}")
        sd.play(test_tone, samplerate=fs, device=device_index)
        sd.wait()
        print("Done!")
    except (ValueError, IndexError) as e:
        print(f"Invalid device index. Please try again.")