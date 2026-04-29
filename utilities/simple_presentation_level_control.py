
# Simple Presentation Level Control
# Automatically adjusts system volume to calibrate presentation levels

import numpy as np
import sounddevice as sd
import time
from threading import Thread, Event
import sys

# Windows audio control
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    PYCAW_AVAILABLE = True
except ImportError:
    print("⚠ WARNING: pycaw not installed. Install with: pip install pycaw")
    PYCAW_AVAILABLE = False

# ============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS
# ============================================================================

# Audio device indices (adjust these to match your system)
headphone_index = 5  # Your headphone output device
loudspeaker_index = 6  # Your loudspeaker output device
in_ear_mic_index = 3  # Your in-ear microphone input device

# Select which device to use: 'headphone' or 'loudspeaker'
ACTIVE_DEVICE = 'headphone'  # Change to 'loudspeaker' to use speakers

# Audio parameters
SAMPLE_RATE = 48000
BLOCK_SIZE = 2048
REFERENCE_PRESSURE = 20e-6  # 20 micropascals (reference for dB SPL)

# Pure tone frequency (Hz)
TONE_FREQUENCY = 1000  # 1 kHz pure tone (adjust as needed)

# Target SPL for calibration (dB)
TARGET_SPL = 40  # Adjust system volume to reach this level for both devices
SPL_TOLERANCE = 1.0  # Acceptable tolerance in dB (within ±1 dB of target)

# Automatic calibration settings
AUTO_CALIBRATE = False  # Set to True to enable automatic volume control (requires pycaw)
MAX_CALIBRATION_TIME = 60  # Maximum seconds to spend calibrating each device
VOLUME_STEP = 0.02  # How much to adjust volume each iteration (0.02 = 2%)

# IMPORTANT: Output amplitude is FIXED at -6 dBFS
# Do NOT change the amplitude in code - adjust your system volume instead!
OUTPUT_AMPLITUDE = 0.5  # Fixed at -6 dBFS (0.5 = 50% of max = -6.02 dB)

# ============================================================================

# Global variables for monitoring
current_spl = 0.0
current_spl_global = 0.0  # For auto-calibration access
is_running = Event()
stop_playback = Event()
calibration_complete = Event()

def generate_pure_tone(frequency=1000, duration_seconds=60, sample_rate=48000, amplitude=OUTPUT_AMPLITUDE):
    """
    Generate a pure tone (sine wave) at FIXED amplitude
    
    Args:
        frequency: Frequency of the tone in Hz
        duration_seconds: Duration of tone in seconds
        sample_rate: Sample rate in Hz
        amplitude: Fixed output amplitude (DO NOT CHANGE - adjust system volume instead!)
        
    Returns:
        Pure tone signal as numpy array (stereo)
    """
    # Generate time array
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate
    
    # Generate sine wave
    tone_mono = np.sin(2 * np.pi * frequency * t)
    
    # Create stereo version (duplicate to both channels)
    tone_stereo = np.column_stack([tone_mono, tone_mono])
    
    # Apply FIXED amplitude (DO NOT CHANGE THIS - calibrate your system volume instead!)
    tone_stereo = tone_stereo * amplitude
    
    return tone_stereo

def get_device_volume_control(device_name_substring):
    """
    Get volume control interface for a specific audio device
    
    Args:
        device_name_substring: Part of the device name to search for
        
    Returns:
        IAudioEndpointVolume interface or None
    """
    if not PYCAW_AVAILABLE:
        return None
    
    try:
        devices = AudioUtilities.GetAllDevices()
        for device in devices:
            if device_name_substring.lower() in device.FriendlyName.lower():
                interface = device.id
                # Get the audio endpoint
                devices_enum = AudioUtilities.GetSpeakers()
                endpoint = devices_enum.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = endpoint.QueryInterface(IAudioEndpointVolume)
                return volume
        return None
    except Exception as e:
        print(f"Error getting device volume control: {e}")
        return None

def set_device_volume(volume_control, volume_level):
    """
    Set device volume (0.0 to 1.0)
    
    Args:
        volume_control: IAudioEndpointVolume interface
        volume_level: Volume level (0.0 = mute, 1.0 = max)
    """
    if volume_control:
        try:
            volume_control.SetMasterVolumeLevelScalar(volume_level, None)
        except Exception as e:
            print(f"Error setting volume: {e}")

def get_device_volume(volume_control):
    """
    Get current device volume (0.0 to 1.0)
    
    Args:
        volume_control: IAudioEndpointVolume interface
        
    Returns:
        Current volume level (0.0 to 1.0)
    """
    if volume_control:
        try:
            return volume_control.GetMasterVolumeLevelScalar()
        except Exception as e:
            print(f"Error getting volume: {e}")
            return 0.5
    return 0.5

def auto_calibrate_volume(volume_control, target_spl, tolerance, max_time):
    """
    Automatically adjust device volume to reach target SPL
    
    Args:
        volume_control: IAudioEndpointVolume interface
        target_spl: Target SPL in dB
        tolerance: Acceptable tolerance in dB
        max_time: Maximum calibration time in seconds
        
    Returns:
        True if calibration successful, False otherwise
    """
    if not volume_control:
        print("⚠ Volume control not available - adjust manually")
        return False
    
    print(f"\n🔧 Auto-calibrating to {target_spl} ± {tolerance} dB SPL...")
    
    start_time = time.time()
    iteration = 0
    
    while (time.time() - start_time) < max_time:
        iteration += 1
        
        # Wait for SPL reading to stabilize
        time.sleep(0.5)
        
        current_spl = current_spl_global
        current_volume = get_device_volume(volume_control)
        
        # Calculate error
        spl_error = target_spl - current_spl
        
        # Check if within tolerance
        if abs(spl_error) <= tolerance:
            print(f"\n✓ Calibration complete! SPL: {current_spl:.1f} dB, Volume: {current_volume*100:.0f}%")
            return True
        
        # Adjust volume proportionally to error
        # Rough estimate: 6 dB SPL change per doubling of volume
        volume_change = spl_error / 20.0  # More conservative adjustment
        new_volume = np.clip(current_volume + volume_change * VOLUME_STEP, 0.0, 1.0)
        
        set_device_volume(volume_control, new_volume)
        
        print(f"  Iteration {iteration}: SPL {current_spl:.1f} dB → Volume {new_volume*100:.0f}% (error: {spl_error:+.1f} dB)")
    
    print(f"\n⚠ Calibration timeout after {max_time}s")
    return False

def calculate_spl(audio_block, reference=REFERENCE_PRESSURE):
    """
    Calculate SPL in dB from audio block
    
    Args:
        audio_block: Audio samples from microphone
        reference: Reference pressure (20 micropascals for SPL)
        
    Returns:
        SPL in dB
    """
    global current_spl_global
    
    # Calculate RMS (root mean square) of the audio block
    rms = np.sqrt(np.mean(audio_block**2))
    
    # Avoid log of zero
    if rms < 1e-10:
        spl = 0.0
    else:
        # Convert to dB SPL
        # Assuming microphone has 1:1 calibration (adjust if needed)
        # For proper calibration, you'd need to know your mic's sensitivity
        spl = 20 * np.log10(rms / reference)
    
    # Update global for auto-calibration
    current_spl_global = spl
    
    return spl

def audio_callback(indata, outdata, frames, time_info, status):
    """
    Callback function for simultaneous playback and recording
    """
    global current_spl
    
    if status:
        print(f"Status: {status}")
    
    # Calculate SPL from input (microphone)
    current_spl = calculate_spl(indata[:, 0])  # Use first channel
    
    # Output is handled by the pink noise stream

def monitor_spl_display():
    """
    Display SPL levels in real-time
    """
    print("\nMonitoring SPL... (Press SPACE to stop)")
    print("-" * 50)
    
    while is_running.is_set() and not stop_playback.is_set():
        # Display current SPL
        bars = int(current_spl / 2)  # Scale for display (2 dB per bar)
        bar_display = "█" * max(0, bars)
        
        print(f"\rSPL: {current_spl:6.1f} dB SPL {bar_display:50s}", end='', flush=True)
        time.sleep(0.1)  # Update 10 times per second

def wait_for_spacebar():
    """
    Wait for spacebar press in a separate thread
    """
    print("\n\nPress SPACE to stop playback...")
    while not stop_playback.is_set():
        try:
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':
                        stop_playback.set()
                        break
            else:
                # Unix/Linux - use select
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == ' ':
                        stop_playback.set()
                        break
        except:
            pass
        time.sleep(0.1)

def play_and_monitor(device_name, output_device, device_name_for_volume=None):
    """
    Play pure tone through specified device and monitor SPL continuously
    Optionally auto-calibrate volume to reach target SPL
    
    Args:
        device_name: Name for display (e.g., "Headphone" or "Loudspeaker")
        output_device: Device index for output
        device_name_for_volume: Substring of device name for volume control (e.g., "Headphones")
    """
    global current_spl, current_spl_global
    current_spl = 0.0
    current_spl_global = 0.0
    is_running.set()
    stop_playback.clear()
    calibration_complete.clear()
    
    print(f"\n{'='*70}")
    print(f"Playing {TONE_FREQUENCY} Hz pure tone via {device_name}")
    print(f"Output device: {output_device}")
    print(f"Input device (mic): {in_ear_mic_index}")
    if AUTO_CALIBRATE and PYCAW_AVAILABLE:
        print(f"Mode: AUTO-CALIBRATION to {TARGET_SPL} dB SPL")
    else:
        print(f"Mode: MANUAL (adjust system volume to reach {TARGET_SPL} dB SPL)")
    print(f"{'='*70}")
    
    # Generate long pure tone (60 seconds, will loop if needed)
    print(f"Generating {TONE_FREQUENCY} Hz pure tone...")
    pure_tone = generate_pure_tone(TONE_FREQUENCY, 60, SAMPLE_RATE)
    
    # Get volume control interface
    volume_control = None
    if AUTO_CALIBRATE and PYCAW_AVAILABLE and device_name_for_volume:
        print(f"Getting volume control for '{device_name_for_volume}'...")
        volume_control = get_device_volume_control(device_name_for_volume)
        if volume_control:
            initial_volume = get_device_volume(volume_control)
            print(f"✓ Volume control acquired. Current volume: {initial_volume*100:.0f}%")
        else:
            print(f"⚠ Could not get volume control - will use manual mode")
    
    # Start monitoring display thread
    display_thread = Thread(target=monitor_spl_display)
    display_thread.daemon = True
    display_thread.start()
    
    # Start spacebar monitoring thread (for manual mode or early stop)
    spacebar_thread = Thread(target=wait_for_spacebar)
    spacebar_thread.daemon = True
    spacebar_thread.start()
    
    try:
        # Create input stream for monitoring
        input_stream = sd.InputStream(
            device=in_ear_mic_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=lambda indata, frames, time_info, status: 
                globals().update({'current_spl': calculate_spl(indata[:, 0])})
        )
        
        # Start input stream
        input_stream.start()
        
        # Start playback
        sd.play(pure_tone, samplerate=SAMPLE_RATE, device=output_device, loop=True)
        
        # Wait for audio to start and SPL to stabilize
        time.sleep(2)
        
        # Auto-calibrate if enabled
        if AUTO_CALIBRATE and volume_control:
            success = auto_calibrate_volume(volume_control, TARGET_SPL, SPL_TOLERANCE, MAX_CALIBRATION_TIME)
            if success:
                calibration_complete.set()
                print(f"\n✓ Auto-calibration successful!")
                print(f"  Press SPACE to continue to next phase...")
            else:
                print(f"\n⚠ Auto-calibration did not converge")
                print(f"  Current SPL: {current_spl_global:.1f} dB (target: {TARGET_SPL} dB)")
                print(f"  Press SPACE when ready to continue...")
        else:
            print(f"\nManual mode: Adjust system volume to reach {TARGET_SPL} dB SPL")
            print(f"Press SPACE when calibrated...")
        
        # Wait for spacebar
        while not stop_playback.is_set():
            time.sleep(0.1)
        
        # Stop playback
        sd.stop()
        
        # Stop input stream
        input_stream.stop()
        input_stream.close()
        
    except KeyboardInterrupt:
        print("\n\nPlayback stopped by user (Ctrl+C)")
        sd.stop()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_running.clear()
        stop_playback.set()
        time.sleep(0.2)  # Let display thread finish
        
        # Show final calibration result
        final_volume = get_device_volume(volume_control) if volume_control else None
        print(f"\n✓ {device_name} calibration complete!")
        if final_volume is not None:
            print(f"  Final volume: {final_volume*100:.0f}%")
        print(f"  Final SPL: {current_spl_global:.1f} dB")

def main():
    """
    Main function - play pure tone and monitor SPL
    First through headphones, then through loudspeakers
    
    CALIBRATION WORKFLOW:
    1. Headphones play at FIXED amplitude → observe SPL
    2. Adjust SYSTEM HEADPHONE VOLUME to reach target SPL
    3. Loudspeakers play at SAME FIXED amplitude → observe SPL  
    4. Adjust SYSTEM LOUDSPEAKER VOLUME to reach same target SPL
    5. Done! Both devices now produce equal SPL for the same digital signal
    """
    print("=" * 70)
    print("PRESENTATION LEVEL CALIBRATION")
    print("=" * 70)
    
    print(f"\n⚠ IMPORTANT INSTRUCTIONS:")
    print(f"  1. The signal amplitude is FIXED at {OUTPUT_AMPLITUDE:.2f} ({20*np.log10(OUTPUT_AMPLITUDE):.1f} dBFS)")
    print(f"  2. Your target SPL is: {TARGET_SPL} dB")
    print(f"  3. Adjust your SYSTEM VOLUME (not this script) to reach target SPL")
    print(f"  4. Once calibrated, both devices will maintain equal presentation levels")
    print()
    
    print(f"\nConfiguration:")
    print(f"  Pure tone frequency: {TONE_FREQUENCY} Hz")
    print(f"  Fixed output amplitude: {OUTPUT_AMPLITUDE:.3f} ({20*np.log10(OUTPUT_AMPLITUDE):.1f} dBFS)")
    print(f"  Target SPL: {TARGET_SPL} dB")
    print(f"  Headphone device: index {headphone_index}")
    print(f"  Loudspeaker device: index {loudspeaker_index}")
    print(f"  Microphone: index {in_ear_mic_index}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print("=" * 70)
    
    # Phase 1: Headphones
    print("\n\n*** PHASE 1: HEADPHONES ***")
    if AUTO_CALIBRATE and PYCAW_AVAILABLE:
        print(f"→ Auto-adjusting HEADPHONE VOLUME to {TARGET_SPL} dB SPL...")
    else:
        print(f"→ Manually adjust HEADPHONE VOLUME to reach {TARGET_SPL} dB SPL")
    play_and_monitor("Headphones", headphone_index, device_name_for_volume="Headphones (Realtek")
    
    # Wait a moment between phases
    time.sleep(1)
    
    # Phase 2: Loudspeakers
    print("\n\n*** PHASE 2: LOUDSPEAKERS ***")
    if AUTO_CALIBRATE and PYCAW_AVAILABLE:
        print(f"→ Auto-adjusting LOUDSPEAKER VOLUME to {TARGET_SPL} dB SPL...")
    else:
        print(f"→ Manually adjust LOUDSPEAKER VOLUME to reach {TARGET_SPL} dB SPL")
    play_and_monitor("Loudspeakers", loudspeaker_index, device_name_for_volume="Q M20")
    
    print("\n\n" + "=" * 70)
    print("✓ Calibration complete!")
    print(f"  Both devices should now produce {TARGET_SPL} dB SPL")
    print(f"  at the fixed signal level of {20*np.log10(OUTPUT_AMPLITUDE):.1f} dBFS")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        is_running.clear()
        print("\n\nInterrupted by user. Exiting...")

