# Check Input/Output Device Index
# Displays available audio input/output devices, allows selection, and tests recording/playback

import sounddevice as sd
import soundfile as sf
import numpy as np
import time

# Audio parameters
SAMPLE_RATE = 48000
INPUT_DURATION = 10  # seconds
OUTPUT_DURATION = 1.0  # seconds


def list_input_devices():
    """Display all available audio input devices."""
    print("\n" + "=" * 70)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("=" * 70)

    devices = sd.query_devices()
    input_devices = []

    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append(idx)
            print(f"\nIndex: {idx}")
            print(f"  Name: {device['name']}")
            print(f"  Input channels: {device['max_input_channels']}")
            print(f"  Sample rate: {device['default_samplerate']} Hz")

    print("\n" + "=" * 70)
    return input_devices


def list_output_devices():
    """Display all available audio output devices."""
    print("\n" + "=" * 70)
    print("AVAILABLE AUDIO OUTPUT DEVICES")
    print("=" * 70)

    devices = sd.query_devices()
    output_devices = []

    for idx, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices.append(idx)
            print(f"\nIndex: {idx}")
            print(f"  Name: {device['name']}")
            print(f"  Output channels: {device['max_output_channels']}")
            print(f"  Sample rate: {device['default_samplerate']} Hz")

    print("\n" + "=" * 70)
    return output_devices


def record_audio(device_index, duration=INPUT_DURATION, sample_rate=SAMPLE_RATE):
    """Record audio from a specified input device."""
    print(f"\nRecording from index {device_index}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print("\nRecording in 3 seconds...")

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("RECORDING NOW... (Make some noise!)")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=device_index,
        dtype='float32'
    )
    sd.wait()

    print("Recording complete!")
    return recording


def play_audio(audio, sample_rate=SAMPLE_RATE, device_index=None):
    """Play audio through a specified output device or the default output device."""
    target = f"device {device_index}" if device_index is not None else "the default output device"
    print(f"\nPlaying audio through {target}...")
    print("Playback starting...")
    sd.play(audio, samplerate=sample_rate, device=device_index)
    sd.wait()
    print("Playback complete!")


def make_test_tone(duration=OUTPUT_DURATION, sample_rate=SAMPLE_RATE, frequency=440, amplitude=0.3):
    """Generate a simple sine wave test tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def analyze_recording(audio):
    """Display basic statistics about the recording."""
    print("\n" + "-" * 70)
    print("RECORDING ANALYSIS")
    print("-" * 70)

    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)

    print(f"Peak level: {peak:.6f} ({peak_db:.1f} dBFS)")
    print(f"RMS level:  {rms:.6f} ({rms_db:.1f} dBFS)")
    print(f"Duration:   {len(audio) / SAMPLE_RATE:.1f} seconds")

    if peak < 0.001:
        print("\nWARNING: Signal is very quiet! Check:")
        print("  - Microphone is not muted")
        print("  - Correct input device selected")
        print("  - Microphone gain/volume is turned up")
    elif peak > 0.99:
        print("\nWARNING: Signal is clipping! Reduce input gain.")
    else:
        print("\nSignal level looks good!")

    print("-" * 70)


def test_input_device():
    """Run the input device checking workflow."""
    input_devices = list_input_devices()

    if not input_devices:
        print("\nNo input devices found!")
        return

    while True:
        print("\nEnter input device index to test (or 'b' to go back, 'q' to quit):")
        choice = input("Input device index: ").strip()

        if choice.lower() == 'q':
            raise KeyboardInterrupt
        if choice.lower() == 'b':
            return

        try:
            device_index = int(choice)
            if device_index not in input_devices:
                print(f"\nInvalid input device index: {device_index}")
                print(f"Please select from: {input_devices}")
                continue

            device_info = sd.query_devices(device_index)
            print(f"\nSelected input device: {device_info['name']}")

            recording = record_audio(device_index, duration=INPUT_DURATION, sample_rate=SAMPLE_RATE)
            analyze_recording(recording)

            playback = input("\nPlay back the recording through the default output device? (y/n): ").strip().lower()
            if playback == 'y':
                play_audio(recording, sample_rate=SAMPLE_RATE)

            save = input("\nSave recording to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"test_recording_device_{device_index}.wav"
                sf.write(filename, recording, SAMPLE_RATE)
                print(f"Saved to: {filename}")

            again = input("\nTest another input device? (y/n): ").strip().lower()
            if again != 'y':
                return

        except ValueError:
            print(f"\nInvalid input: '{choice}'")
            print("Please enter a numeric device index, 'b', or 'q'")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def test_output_device():
    """Run the output device checking workflow."""
    output_devices = list_output_devices()

    if not output_devices:
        print("\nNo output devices found!")
        return

    test_tone = make_test_tone()

    while True:
        print("\nEnter output device index to test (or 'b' to go back, 'q' to quit):")
        choice = input("Output device index: ").strip()

        if choice.lower() == 'q':
            raise KeyboardInterrupt
        if choice.lower() == 'b':
            return

        try:
            device_index = int(choice)
            if device_index not in output_devices:
                print(f"\nInvalid output device index: {device_index}")
                print(f"Please select from: {output_devices}")
                continue

            device_info = sd.query_devices(device_index)
            print(f"\nSelected output device: {device_info['name']}")
            play_audio(test_tone, sample_rate=SAMPLE_RATE, device_index=device_index)

            again = input("\nTest another output device? (y/n): ").strip().lower()
            if again != 'y':
                return

        except ValueError:
            print(f"\nInvalid input: '{choice}'")
            print("Please enter a numeric device index, 'b', or 'q'")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function."""
    print("\nAudio Device Checker")
    print("This tool helps you find and test microphone inputs and speaker outputs")

    while True:
        print("\nChoose an option:")
        print("  1 - Check input devices")
        print("  2 - Check output devices")
        print("  3 - Check both")
        print("  q - Quit")

        choice = input("Selection: ").strip().lower()

        if choice == '1':
            test_input_device()
        elif choice == '2':
            test_output_device()
        elif choice == '3':
            test_input_device()
            test_output_device()
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, or q.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")

