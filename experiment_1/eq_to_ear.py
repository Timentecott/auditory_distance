# -*- coding: utf-8 -*-
#play a calibration tone through the selected output device
#whilst calibration tone is playing, record the sound using the selected input device
#adjust playback to read 65dB SPL at the microphone position
import sounddevice as sd	
import numpy as np
import scipy.signal as signal
import time
import json
import threading
from pathlib import Path

# Configuration
TARGET_SPL_DB = 50.0  # Target SPL in dB
REF_PRESSURE = 20e-6  # Reference pressure (20 μPa)
CALIBRATION_FREQ = 1000  # Hz (1 kHz calibration tone)
DURATION = 2.0  # seconds
SAMPLE_RATE = 48000

# Microphone sensitivity calibration
# Adjust this based on your microphone calibration (dB re 1V/Pa)
# For most measurement mics this is around -40 to -30 dBV/Pa
MIC_SENSITIVITY_DBV_PA = -46.0  # 46 for charlotte's mic

def generate_calibration_tone(freq=CALIBRATION_FREQ, duration=DURATION, fs=SAMPLE_RATE, amplitude=0.1):
    """Generate a sine wave calibration tone."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    return tone.astype(np.float32)

def calculate_spl(audio_rms, mic_sensitivity_dbv_pa=MIC_SENSITIVITY_DBV_PA):
    """
    Calculate SPL from RMS voltage assuming microphone sensitivity.
    
    Args:
        audio_rms: RMS voltage from microphone (normalized 0-1 scale)
        mic_sensitivity_dbv_pa: Microphone sensitivity in dBV/Pa
    
    Returns:
        SPL in dB SPL
    """
    # Validate input
    if not np.isfinite(audio_rms) or audio_rms <= 0:
        return float('nan')
    
    # Convert normalized RMS to voltage (assuming 1.0 = 1V for simplicity)
    voltage = audio_rms
    
    # Convert mic sensitivity to linear (V/Pa)
    mic_sens_linear = 10 ** (mic_sensitivity_dbv_pa / 20.0)
    
    # Calculate pressure in Pa
    pressure_pa = voltage / mic_sens_linear
    
    # Avoid log of zero or negative
    if pressure_pa <= 0:
        return float('nan')
    
    # Calculate SPL
    spl_db = 20 * np.log10(pressure_pa / REF_PRESSURE)
    return spl_db

def calibrate_output(output_device, input_device, target_spl=TARGET_SPL_DB, max_iterations=10):
    """
    Calibrate output level to achieve target SPL at microphone.
    
    Args:
        output_device: Output device index
        input_device: Input device index
        target_spl: Target SPL in dB
        max_iterations: Maximum calibration attempts
    
    Returns:
        dict with calibration results
    """
    print(f"\n=== Calibration Start ===")
    print(f"Output device: {output_device}")
    print(f"Input device: {input_device}")
    print(f"Target SPL: {target_spl} dB")
    
    amplitude = 0.1  # Initial amplitude
    calibration_data = []
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        print(f"  Current amplitude: {amplitude:.4f} ({20*np.log10(amplitude):.1f} dBFS)")
        
        # Generate tone
        tone = generate_calibration_tone(amplitude=amplitude)
        
        print("  Starting simultaneous playback and recording...")
        
        # Use threading with proper synchronization
        recorded = None
        record_done = threading.Event()
        play_started = threading.Event()
        
        def record_thread():
            nonlocal recorded
            # Wait for playback to start
            play_started.wait(timeout=2.0)
            # Small delay to sync
            time.sleep(0.05)
            recorded = sd.rec(len(tone), samplerate=SAMPLE_RATE, 
                            channels=1, device=input_device, blocking=True)
            record_done.set()
        
        def play_thread():
            play_started.set()
            sd.play(tone, samplerate=SAMPLE_RATE, device=output_device, blocking=True)
        
        # Start both threads
        rec_thread = threading.Thread(target=record_thread)
        play_thd = threading.Thread(target=play_thread)
        
        rec_thread.start()
        time.sleep(0.05)  # Ensure recording thread is ready
        play_thd.start()
        
        # Wait for both to finish
        play_thd.join(timeout=DURATION + 2.0)
        record_done.wait(timeout=DURATION + 2.0)
        rec_thread.join()
        
        # Validate recording
        if recorded is None or len(recorded) == 0:
            print("  Error: No audio recorded, retrying...")
            continue
        
        # Calculate RMS of recorded signal (ignore first/last 0.5s to avoid transients)
        trim_samples = int(0.5 * SAMPLE_RATE)
        if len(recorded) <= 2 * trim_samples:
            print("  Error: Recording too short")
            continue
            
        signal_trim = recorded[trim_samples:-trim_samples]
        
        # Clip extreme values to prevent overflow
        signal_trim = np.clip(signal_trim, -1.0, 1.0)
        
        rms = np.sqrt(np.mean(signal_trim ** 2))
        
        # Handle edge cases
        if rms < 1e-10 or not np.isfinite(rms):
            print(f"  Error: Invalid RMS value ({rms}), skipping iteration")
            amplitude *= 2.0  # Try increasing amplitude
            continue
        
        # Calculate SPL
        measured_spl = calculate_spl(rms)
        
        print(f"  Measured RMS: {rms:.6f}")
        print(f"  Measured SPL: {measured_spl:.1f} dB")
        
        calibration_data.append({
            'iteration': iteration + 1,
            'amplitude': float(amplitude),
            'rms': float(rms),
            'spl_db': float(measured_spl)
        })
        
        # Check if within +/-1 dB
        error = target_spl - measured_spl
        if abs(error) < 1.0:
            print(f"\nCalibration successful! SPL within +/-1 dB of target.")
            break
        
        # Adjust amplitude (SPL is in log scale, so linear adjustment)
        # error in dB -> amplitude adjustment
        amplitude_adjustment_db = error
        amplitude *= 10 ** (amplitude_adjustment_db / 20.0)
        
        # Safety limit
        if amplitude > 0.5:
            print(f"Warning: amplitude capped at 0.5 for safety")
            amplitude = 0.5
        
        if iteration == max_iterations - 1:
            print(f"\nMax iterations reached. Final error: {error:.1f} dB")
    
    result = {
        'success': abs(error) < 1.0,
        'final_amplitude': float(amplitude),
        'final_spl_db': float(measured_spl),
        'target_spl_db': target_spl,
        'error_db': float(error),
        'iterations': calibration_data,
        'output_device': output_device,
        'input_device': input_device,
        'mic_sensitivity_dbv_pa': MIC_SENSITIVITY_DBV_PA
    }
    
    return result

def list_devices():
    """Print available audio devices."""
    print("\n=== Available Audio Devices ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']}")
        print(f"   In: {dev['max_input_channels']}, Out: {dev['max_output_channels']}, SR: {dev['default_samplerate']}")
    print()

def save_calibration(result, filename='calibration.json'):
    """Save calibration result to JSON."""
    filepath = Path(filename)
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nCalibration saved to {filepath}")

def main():
    """Main calibration routine."""
    print("=== Audio Calibration Tool ===")
    
    # List devices
    list_devices()
    
    # Get device indices from user
    try:
        output_device = int(input("Enter output device index (loudspeaker/headphone): "))
        input_device = int(input("Enter input device index (microphone): "))
    except ValueError:
        print("Invalid device index")
        return
    
    # Optional: adjust target SPL
    use_default = input(f"Use default target SPL ({TARGET_SPL_DB} dB)? [Y/n]: ").strip().lower()
    if use_default == 'n':
        try:
            target_spl = float(input("Enter target SPL (dB): "))
        except ValueError:
            print("Invalid SPL, using default")
            target_spl = TARGET_SPL_DB
    else:
        target_spl = TARGET_SPL_DB
    
    # Run calibration
    result = calibrate_output(output_device, input_device, target_spl=target_spl)
    
    # Print summary
    print("\n=== Calibration Summary ===")
    print(f"Success: {result['success']}")
    print(f"Final amplitude: {result['final_amplitude']:.4f} ({20*np.log10(result['final_amplitude']):.1f} dBFS)")
    print(f"Final SPL: {result['final_spl_db']:.1f} dB")
    print(f"Target SPL: {result['target_spl_db']:.1f} dB")
    print(f"Error: {result['error_db']:.1f} dB")
    
    # Save
    save_cal = input("\nSave calibration? [Y/n]: ").strip().lower()
    if save_cal != 'n':
        filename = input("Filename [calibration.json]: ")

        if not filename:
            filename = 'calibration.json'
        save_calibration(result, filename)

if __name__ == '__main__':
    main()
