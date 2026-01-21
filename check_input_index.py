# Check Input Device Index
# Displays available audio input devices, allows selection, and tests recording/playback

import sounddevice as sd
import soundfile as sf
import numpy as np
import time

# Audio parameters
SAMPLE_RATE = 48000
DURATION = 10  # seconds

def display_audio_devices():
    """Display all available audio input devices"""
    print("\n" + "=" * 70)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("=" * 70)
    
    devices = sd.query_devices()
    input_devices = []
    
    for idx, device in enumerate(devices):
        # Check if device has input channels
        if device['max_input_channels'] > 0:
            input_devices.append(idx)
            print(f"\nIndex: {idx}")
            print(f"  Name: {device['name']}")
            print(f"  Input channels: {device['max_input_channels']}")
            print(f"  Sample rate: {device['default_samplerate']} Hz")
    
    print("\n" + "=" * 70)
    return input_devices

def record_audio(device_index, duration=10, sample_rate=48000):
    """
    Record audio from specified device
    
    Args:
        device_index: Index of input device
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Recorded audio as numpy array
    """
    print(f"\nRecording from index {device_index}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print("\nRecording in 3 seconds...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("?? RECORDING NOW... (Make some noise!)")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=device_index,
        dtype='float32'
    )
    
    # Wait for recording to complete
    sd.wait()
    
    print("? Recording complete!")
    
    return recording

def play_audio(audio, sample_rate=48000):
    """
    Play audio through default output device
    
    Args:
        audio: Audio array to play
        sample_rate: Sample rate in Hz
    """
    print("\nPlaying back recorded audio...")
    print("?? Playback starting...")
    
    # Play audio
    sd.play(audio, samplerate=sample_rate)
    
    # Wait for playback to complete
    sd.wait()
    
    print("? Playback complete!")

def analyze_recording(audio):
    """Display basic statistics about the recording"""
    print("\n" + "-" * 70)
    print("RECORDING ANALYSIS")
    print("-" * 70)
    
    # Calculate RMS level
    rms = np.sqrt(np.mean(audio**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Calculate peak level
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)
    
    print(f"Peak level: {peak:.6f} ({peak_db:.1f} dBFS)")
    print(f"RMS level:  {rms:.6f} ({rms_db:.1f} dBFS)")
    print(f"Duration:   {len(audio) / SAMPLE_RATE:.1f} seconds")
    
    # Check if signal is too quiet
    if peak < 0.001:
        print("\n? WARNING: Signal is very quiet! Check:")
        print("  - Microphone is not muted")
        print("  - Correct input device selected")
        print("  - Microphone gain/volume is turned up")
    elif peak > 0.99:
        print("\n? WARNING: Signal is clipping! Reduce input gain.")
    else:
        print("\n? Signal level looks good!")
    
    print("-" * 70)

def main():
    """Main function"""
    print("\nAudio Input Device Checker")
    print("This tool helps you find and test microphone input devices")
    
    while True:
        # Display available input devices
        input_devices = display_audio_devices()
        
        if not input_devices:
            print("\n? No input devices found!")
            return
        
        # Get user selection
        print("\nEnter device index to test (or 'q' to quit):")
        choice = input("Device index: ").strip()
        
        if choice.lower() == 'q':
            print("Exiting...")
            break
        
        try:
            device_index = int(choice)
            
            # Validate selection
            if device_index not in input_devices:
                print(f"\n? Invalid device index: {device_index}")
                print(f"Please select from: {input_devices}")
                continue
            
            # Get device info
            device_info = sd.query_devices(device_index)
            print(f"\n? Selected device: {device_info['name']}")
            
            # Record audio
            recording = record_audio(device_index, duration=DURATION, sample_rate=SAMPLE_RATE)
            
            # Analyze recording
            analyze_recording(recording)
            
            # Ask to play back
            playback = input("\nPlay back recording? (y/n): ").strip().lower()
            if playback == 'y':
                play_audio(recording, sample_rate=SAMPLE_RATE)
            
            # Ask to save
            save = input("\nSave recording to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"test_recording_device_{device_index}.wav"
                sf.write(filename, recording, SAMPLE_RATE)
                print(f"? Saved to: {filename}")
            
            # Ask to test another device
            again = input("\nTest another device? (y/n): ").strip().lower()
            if again != 'y':
                print("Exiting...")
                break
            
        except ValueError:
            print(f"\n? Invalid input: '{choice}'")
            print("Please enter a numeric device index or 'q' to quit")
        except Exception as e:
            print(f"\n? Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")

