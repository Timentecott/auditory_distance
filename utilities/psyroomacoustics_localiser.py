# -*- coding: utf-8 -*-
# Room Impulse Response (RIR) Generator using pyroomacoustics
# Applies realistic room acoustics to audio files

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from pathlib import Path

# ============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS
# ============================================================================

# Input/Output files
INPUT_AUDIO = r'C:\Users\tim_e\source\repos\auditory_distance\original_audios\environment\blackbird_bbc_01.wav'
OUTPUT_AUDIO = r'C:\Users\tim_e\source\repos\auditory_distance\brir_localised_stimuli\environment\blackbird_bbc_01_room.wav'

# Room dimensions (in meters)
ROOM_WIDTH = 5.0    # x dimension (meters)
ROOM_LENGTH = 6.0   # y dimension (meters)
ROOM_HEIGHT = 3.0   # z dimension (meters)

# Room acoustic properties
ABSORPTION = 0.3    # Wall absorption coefficient (0=reflective, 1=absorptive)
MAX_ORDER = 15      # Maximum reflection order (higher = more reverb, slower)

# Source position (x, y, z) in meters
SOURCE_POSITION = np.array([0.5, 5, 1.2])  # Center-ish of room, ear height

# Microphone position (x, y, z) in meters - listener location
MIC_POSITION = np.array([1, 4, 1.2])     # 1.5m away from source

# Visualization
SHOW_PLOT = True    # Set to False to skip visualization
SAVE_PLOT = True    # Save room layout plot

# ============================================================================

def create_room_with_rir(room_dim, source_pos, mic_pos, audio_signal, fs, 
                         absorption=0.3, max_order=15):
    """
    Create a 3D room and generate RIR
    
    Args:
        room_dim: [width, length, height] in meters
        source_pos: [x, y, z] source position in meters
        mic_pos: [x, y, z] microphone position in meters
        audio_signal: Input audio signal
        fs: Sample rate
        absorption: Wall absorption coefficient (0-1)
        max_order: Maximum reflection order
        
    Returns:
        room: pyroomacoustics Room object
        reverberant_signal: Audio with RIR applied
    """
    print("\n" + "="*70)
    print("ROOM IMPULSE RESPONSE GENERATOR")
    print("="*70)
    
    print(f"\nRoom Configuration:")
    print(f"  Dimensions: {room_dim[0]}m (W) x {room_dim[1]}m (L) x {room_dim[2]}m (H)")
    print(f"  Volume: {np.prod(room_dim):.1f} m3")
    print(f"  Absorption: {absorption:.2f}")
    print(f"  Max reflection order: {max_order}")
    
    print(f"\nSource position: ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}) m")
    print(f"Mic position: ({mic_pos[0]:.2f}, {mic_pos[1]:.2f}, {mic_pos[2]:.2f}) m")
    
    # Calculate distance
    distance = np.linalg.norm(source_pos - mic_pos)
    print(f"Direct path distance: {distance:.2f} m")
    
    # Create 3D room
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(absorption),
        max_order=max_order
    )
    
    # Add source with audio signal
    room.add_source(source_pos, signal=audio_signal)
    
    # Add microphone (single mic for mono, or array for stereo)
    room.add_microphone(mic_pos)
    
    print("\nSimulating room acoustics...")
    # Compute RIR
    room.compute_rir()
    
    # Get the RIR
    rir = room.rir[0][0]  # First microphone, first source
    print(f"? RIR computed: {len(rir)} samples ({len(rir)/fs*1000:.1f} ms)")
    
    # Analyze RIR
    analyze_rir(rir, fs)
    
    # Simulate (convolve source signal with RIR)
    print("\nApplying RIR to audio signal...")
    room.simulate()
    reverberant_signal = room.mic_array.signals[0, :]
    
    print(f"? Output signal length: {len(reverberant_signal)} samples ({len(reverberant_signal)/fs:.2f}s)")
    
    return room, reverberant_signal, rir

def analyze_rir(rir, fs):
    """Analyze and display RIR characteristics"""
    print("\nRIR Analysis:")
    
    # Find direct sound (first significant peak)
    threshold = 0.01 * np.max(np.abs(rir))
    direct_idx = np.where(np.abs(rir) > threshold)[0][0]
    direct_time = direct_idx / fs * 1000
    
    print(f"  Direct sound arrival: {direct_time:.2f} ms")
    
    # RT60 estimation (rough)
    energy = np.cumsum(rir[::-1]**2)[::-1]
    energy_db = 10 * np.log10(energy / energy[0] + 1e-10)
    
    # Find -60dB point
    try:
        rt60_idx = np.where(energy_db < -60)[0][0]
        rt60 = rt60_idx / fs
        print(f"  Estimated RT60: {rt60:.3f} seconds")
    except:
        print(f"  RT60: Could not estimate (signal too short)")
    
    # Peak level
    peak_db = 20 * np.log10(np.max(np.abs(rir)))
    print(f"  Peak level: {peak_db:.1f} dBFS")

def plot_room_and_rir(room, rir, fs, save_path=None):
    """Visualize the room and RIR"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Room layout (3D view)
    ax1 = fig.add_subplot(131, projection='3d')
    room.plot(img_order=0, ax=ax1)
    ax1.set_title('Room Layout (3D View)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # Plot 2: RIR time domain
    ax2 = fig.add_subplot(132)
    time = np.arange(len(rir)) / fs * 1000
    ax2.plot(time, rir)
    ax2.set_title('Room Impulse Response')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RIR spectrogram
    ax3 = fig.add_subplot(133)
    ax3.specgram(rir, Fs=fs, NFFT=512, noverlap=256, cmap='viridis')
    ax3.set_title('RIR Spectrogram')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_ylim([0, 8000])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"? Plot saved to: {save_path}")
    
    plt.show()

def main():
    """Main function"""
    # Check if input file exists
    input_path = Path(INPUT_AUDIO)
    if not input_path.exists():
        print(f"Error: Input file not found: {INPUT_AUDIO}")
        return
    
    # Load input audio
    print(f"\nLoading input audio: {input_path.name}")
    fs, audio = wavfile.read(INPUT_AUDIO)
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {len(audio)/fs:.2f} seconds")
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        print("  Converting stereo to mono...")
        audio = audio.mean(axis=1)
    
    # Normalize to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float64) / 2147483648.0
    
    # Create room dimensions array
    room_dim = np.array([ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT])
    
    # Generate RIR and apply to audio
    room, reverberant_audio, rir = create_room_with_rir(
        room_dim=room_dim,
        source_pos=SOURCE_POSITION,
        mic_pos=MIC_POSITION,
        audio_signal=audio,
        fs=fs,
        absorption=ABSORPTION,
        max_order=MAX_ORDER
    )
    
    # Normalize output to prevent clipping
    max_val = np.max(np.abs(reverberant_audio))
    if max_val > 0.99:
        print(f"\nNormalizing output (peak: {max_val:.3f})")
        reverberant_audio = reverberant_audio * (0.99 / max_val)
    
    # Save output
    output_path = Path(OUTPUT_AUDIO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert back to int16 for WAV
    output_int16 = (reverberant_audio * 32767).astype(np.int16)
    wavfile.write(OUTPUT_AUDIO, fs, output_int16)
    print(f"\n? Reverberant audio saved to: {OUTPUT_AUDIO}")
    
    # Visualization
    if SHOW_PLOT:
        plot_save_path = OUTPUT_AUDIO.replace('.wav', '_room_plot.png') if SAVE_PLOT else None
        plot_room_and_rir(room, rir, fs, save_path=plot_save_path)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)

if __name__ == "__main__":
    main()