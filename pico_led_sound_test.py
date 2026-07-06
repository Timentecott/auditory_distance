"""
Test script for controlling Raspberry Pi Pico LED while playing sound.
Plays a sound file and flashes an LED on pin 16 of the Pico via serial communication.
"""

import serial
import time
import pygame

# Configuration
PICO_PORT = 'COM3'  # Change to your Pico's port (COM3 on Windows, /dev/ttyACM0 on Linux/Mac)
BAUD_RATE = 115200
SOUND_FILE = r'C:\Users\tim_e\source\repos\auditory_distance\experiment_2\audio_stimuli\pink_noise_48k_30s_300_8000hz.wav'
LED_FLASH_DURATION = 0.5  # Duration of LED flash in seconds
LED_PIN = 16


def setup_serial_connection():
    """
    Initialize serial connection to Pico.
    """
    try:
        ser = serial.Serial(PICO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Pico to initialize
        print(f"Connected to Pico on {PICO_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to Pico: {e}")
        return None


def flash_led(ser, duration=0.5):
    """
    Send command to Pico to flash LED on pin 16.

    Args:
        ser: Serial connection object
        duration: Flash duration in seconds
    """
    if ser is None:
        print("Serial connection not available")
        return

    try:
        # Send command to turn LED on
        ser.write(b'1')  # 1 = LED ON
        time.sleep(duration)
        # Send command to turn LED off
        ser.write(b'0')  # 0 = LED OFF
        print(f"LED flashed for {duration} seconds")
    except serial.SerialException as e:
        print(f"Error sending command to Pico: {e}")


def play_sound_and_flash(ser, sound_file_path, flash_duration=0.5):
    """
    Play sound file and flash LED simultaneously.

    Args:
        ser: Serial connection object
        sound_file_path: Path to the sound file
        flash_duration: Duration of LED flash in seconds
    """
    # Initialize pygame mixer
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file_path)
        print(f"Loaded sound file: {sound_file_path}")
    except Exception as e:
        print(f"Error loading sound file: {e}")
        return

    # Play sound and flash LED
    print("Starting sound and LED flash...")
    pygame.mixer.music.play()
    flash_led(ser, flash_duration)

    # Wait for sound to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    print("Sound and LED flash complete")


def main():
    """
    Main function to run the test.
    """
    # Setup serial connection
    ser = setup_serial_connection()

    if ser is None:
        print("Failed to connect to Pico. Exiting.")
        return

    try:
        # Run the test
        play_sound_and_flash(ser, SOUND_FILE, LED_FLASH_DURATION)

    except KeyboardInterrupt:
        print("Test interrupted by user")

    finally:
        # Stop sound playback
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # Close serial connection
        if ser and ser.is_open:
            ser.close()
            print("Serial connection closed")


if __name__ == '__main__':
    main()
