# View and analyze binaural WAV file with pyfar
# Displays waveforms, spectrograms, and frequency analysis

import numpy as np
import matplotlib.pyplot as plt
import pyfar as pf
from pathlib import Path
from scipy import signal as scipy_signal

# ============================================================================
# CONFIGURATION
# ============================================================================

AUDIO_FILE = r'C:\Users\tim_e\source\repos\auditory_distance\binaural_recordings\ISTS-V1.0_60s_24bit_2_binaural_recording.wav'

# ============================================================================

def view_binaural_audio(filename):
    """
    Load and visualize binaural audio file using pyfar
    
    Args:
        filename: Path to stereo WAV file
    """
    print("\n" + "="*70)
    print("BINAURAL AUDIO ANALYSIS WITH PYFAR")
    print("="*70)
    
    # Check if file exists
    file_path = Path(filename)
    if not file_path.exists():
        print(f"Error: File not found: {filename}")
        return
    
    # Load audio with pyfar
    print(f"\nLoading: {file_path.name}")
    signal = pf.io.read_audio(filename)
    
    # Display basic information
    print("\n" + "-"*70)
    print("AUDIO INFORMATION")
    print("-"*70)
    print(f"  Sample rate: {signal.sampling_rate} Hz")
    print(f"  Duration: {signal.n_samples / signal.sampling_rate:.2f} seconds")
    print(f"  Samples: {signal.n_samples}")
    print(f"  Channels: {signal.cshape[0]}")
    
    if signal.cshape[0] == 2:
        print(f"  Left channel (0): First channel")
        print(f"  Right channel (1): Second channel")
    
    # Calculate levels and shared axis limits
    print("\n" + "-"*70)
    print("SIGNAL LEVELS")
    print("-"*70)
    
    max_amplitude = 0
    min_freq_mag = float('inf')
    max_freq_mag = float('-inf')
    
    for ch in range(signal.cshape[0]):
        channel_data = signal.time[ch, :]
        rms = np.sqrt(np.mean(channel_data**2))
        peak = np.max(np.abs(channel_data))
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Track maximum amplitude for shared y-axis
        max_amplitude = max(max_amplitude, peak)
        
        # Track frequency magnitude range for shared y-axis
        freq_mag = 20 * np.log10(np.abs(signal[ch].freq).flatten() + 1e-10)
        # Only consider frequencies between 20 Hz and 20 kHz
        freqs = signal[ch].frequencies
        valid_mask = (freqs >= 20) & (freqs <= 20000)
        if np.any(valid_mask):
            min_freq_mag = min(min_freq_mag, np.min(freq_mag[valid_mask]))
            max_freq_mag = max(max_freq_mag, np.max(freq_mag[valid_mask]))
        
        ch_name = "Left" if ch == 0 else "Right"
        print(f"  {ch_name} channel:")
        print(f"    Peak: {peak:.6f} ({peak_db:.1f} dBFS)")
        print(f"    RMS:  {rms:.6f} ({rms_db:.1f} dBFS)")
    
    # Set shared axis limits with some padding
    time_ylim = [-max_amplitude * 1.05, max_amplitude * 1.05]
    freq_ylim = [min_freq_mag - 5, max_freq_mag + 5]
    
    print(f"\n  Shared time domain y-axis: {time_ylim[0]:.4f} to {time_ylim[1]:.4f}")
    print(f"  Shared frequency y-axis: {freq_ylim[0]:.1f} to {freq_ylim[1]:.1f} dB")
    
    # Calculate shared color scale for spectrograms using scipy
    print("\n  Calculating spectrogram ranges...")
    
    vmin_list = []
    vmax_list = []
    
    for ch in range(signal.cshape[0]):
        # Calculate spectrogram using scipy
        f, t, Sxx = scipy_signal.spectrogram(
            signal.time[ch, :],
            fs=signal.sampling_rate,
            window='hann',
            nperseg=1024,
            noverlap=512
        )
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        vmin_list.append(np.min(Sxx_db))
        vmax_list.append(np.max(Sxx_db))
    
    vmin = min(vmin_list)
    vmax = max(vmax_list)
    
    print(f"  Shared spectrogram color scale: {vmin:.1f} to {vmax:.1f} dB")
    
    # Create visualizations
    print("\n" + "-"*70)
    print("CREATING VISUALIZATIONS...")
    print("-"*70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Left channel waveform
    ax1 = plt.subplot(3, 2, 1)
    pf.plot.time(signal[0], ax=ax1, unit='s')
    ax1.set_title('Left Ear - Time Domain', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(time_ylim)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Right channel waveform
    ax2 = plt.subplot(3, 2, 2)
    pf.plot.time(signal[1], ax=ax2, unit='s')
    ax2.set_title('Right Ear - Time Domain', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_ylim(time_ylim)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Left channel spectrogram
    ax3 = plt.subplot(3, 2, 3)
    pf.plot.spectrogram(signal[0], ax=ax3, vmin=vmin, vmax=vmax)
    ax3.set_title('Left Ear - Spectrogram', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)')
    
    # Plot 4: Right channel spectrogram
    ax4 = plt.subplot(3, 2, 4)
    pf.plot.spectrogram(signal[1], ax=ax4, vmin=vmin, vmax=vmax)
    ax4.set_title('Right Ear - Spectrogram', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency (Hz)')
    
    # Plot 5: Left channel frequency spectrum
    ax5 = plt.subplot(3, 2, 5)
    pf.plot.freq(signal[0], ax=ax5)
    ax5.set_title('Left Ear - Frequency Spectrum', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude (dB)')
    ax5.set_xlim([20, 20000])
    ax5.set_ylim(freq_ylim)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Right channel frequency spectrum
    ax6 = plt.subplot(3, 2, 6)
    pf.plot.freq(signal[1], ax=ax6)
    ax6.set_title('Right Ear - Frequency Spectrum', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude (dB)')
    ax6.set_xlim([20, 20000])
    ax6.set_ylim(freq_ylim)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_png = filename.replace('.wav', '_analysis.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"\nCHECK Analysis plot saved to: {output_png}")
    
    # Show plot
    plt.show()
    
    # Additional analysis: Interaural Time Difference (ITD)
    print("\n" + "-"*70)
    print("BINAURAL ANALYSIS")
    print("-"*70)
    
    left_data = signal.time[0, :]
    right_data = signal.time[1, :]
    
    # Calculate cross-correlation to estimate ITD
    correlation = np.correlate(left_data, right_data, mode='full')
    max_corr_idx = np.argmax(correlation)
    center = len(correlation) // 2
    lag_samples = max_corr_idx - center
    lag_ms = (lag_samples / signal.sampling_rate) * 1000
    
    print(f"  Estimated ITD (lag): {lag_samples} samples ({lag_ms:.2f} ms)")
    print(f"  Maximum correlation: {correlation[max_corr_idx] / len(left_data):.3f}")
    
    if abs(lag_ms) < 1:
        print("  CHECK Signals are nearly in phase (mono-like)")
    else:
        print(f"  CHECK {'Right' if lag_samples > 0 else 'Left'} ear is leading")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

def main():
    """Main function"""
    view_binaural_audio(AUDIO_FILE)

if __name__ == "__main__":
    main()
