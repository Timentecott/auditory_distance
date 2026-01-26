# Apply Butterworth filters (HPF and/or LPF) to stereo WAV files using pyfar

import pyfar as pf
import numpy as np
from pathlib import Path
from scipy.io import wavfile

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output files
INPUT_WAV = r'C:\Users\tim_e\source\repos\auditory_distance\binaural_recordings\noise\pink_noise.wav'
OUTPUT_WAV = r'C:\Users\tim_e\source\repos\auditory_distance\binaural_recordings\noise\pink_noise_filtered.wav'

# Filter settings
APPLY_HPF = True        # Apply high-pass filter
HPF_CUTOFF = 80         # High-pass filter cutoff frequency in Hz
HPF_ORDER = 4           # High-pass filter order

APPLY_LPF = False       # Apply low-pass filter
LPF_CUTOFF = 6000       # Low-pass filter cutoff frequency in Hz
LPF_ORDER = 4           # Low-pass filter order

# ============================================================================

def apply_butterworth_filters(input_file, output_file, 
                              apply_hpf=True, hpf_cutoff=80, hpf_order=4,
                              apply_lpf=False, lpf_cutoff=8000, lpf_order=4):
    """
    Apply Butterworth high-pass and/or low-pass filters to a WAV file.
    
    Args:
        input_file: Path to input WAV file (mono or stereo)
        output_file: Path to save filtered WAV file
        apply_hpf: Whether to apply high-pass filter
        hpf_cutoff: High-pass filter cutoff frequency in Hz
        hpf_order: High-pass filter order
        apply_lpf: Whether to apply low-pass filter
        lpf_cutoff: Low-pass filter cutoff frequency in Hz
        lpf_order: Low-pass filter order
    """
    print("\n" + "="*70)
    print("BUTTERWORTH FILTER APPLICATION")
    print("="*70)
    
    # Determine filter type description
    if apply_hpf and apply_lpf:
        filter_desc = f"Band-pass filter ({hpf_cutoff} - {lpf_cutoff} Hz)"
    elif apply_hpf:
        filter_desc = f"High-pass filter ({hpf_cutoff} Hz)"
    elif apply_lpf:
        filter_desc = f"Low-pass filter ({lpf_cutoff} Hz)"
    else:
        print("\n? WARNING: No filters enabled! Output will be same as input.")
        filter_desc = "No filtering"
    
    print(f"\nFilter configuration: {filter_desc}")
    
    # Load audio file
    print(f"\nLoading: {Path(input_file).name}")
    fs, audio_data = wavfile.read(input_file)
    
    # Convert to float if integer format
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
        bit_depth = 16
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
        bit_depth = 32
    else:
        audio_data = audio_data.astype(np.float32)
        bit_depth = 32
    
    # Get info
    is_stereo = audio_data.ndim > 1
    n_channels = audio_data.shape[1] if is_stereo else 1
    duration = len(audio_data) / fs
    
    print(f"  Sample rate: {fs} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(audio_data)}")
    print(f"  Bit depth: {bit_depth}")
    
    # Analysis before filtering
    max_before = np.max(np.abs(audio_data))
    rms_before = np.sqrt(np.mean(audio_data**2))
    
    print(f"\nBefore filtering:")
    print(f"  Peak level: {max_before:.6f} ({20*np.log10(max_before + 1e-10):.1f} dBFS)")
    print(f"  RMS level:  {rms_before:.6f} ({20*np.log10(rms_before + 1e-10):.1f} dBFS)")
    
    # Validate filter settings
    nyquist = fs / 2
    if apply_hpf and hpf_cutoff >= nyquist:
        print(f"\n? ERROR: HPF cutoff ({hpf_cutoff} Hz) must be below Nyquist frequency ({nyquist} Hz)")
        return None, None
    if apply_lpf and lpf_cutoff >= nyquist:
        print(f"\n? ERROR: LPF cutoff ({lpf_cutoff} Hz) must be below Nyquist frequency ({nyquist} Hz)")
        return None, None
    if apply_hpf and apply_lpf and hpf_cutoff >= lpf_cutoff:
        print(f"\n? ERROR: HPF cutoff ({hpf_cutoff} Hz) must be below LPF cutoff ({lpf_cutoff} Hz)")
        return None, None
    
    # Apply filters
    print(f"\nApplying Butterworth filter(s):")
    
    if is_stereo:
        # Process each channel separately
        print(f"  Processing stereo (2 channels)...")
        filtered_channels = []
        
        for ch in range(n_channels):
            # Convert to pyfar Signal
            signal = pf.Signal(audio_data[:, ch], fs)
            
            # Apply high-pass filter
            if apply_hpf:
                print(f"    Channel {ch+1}: Applying HPF at {hpf_cutoff} Hz (order {hpf_order})")
                signal = pf.dsp.filter.butterworth(
                    signal, 
                    N=hpf_order, 
                    frequency=hpf_cutoff, 
                    btype='highpass'
                )
            
            # Apply low-pass filter
            if apply_lpf:
                print(f"    Channel {ch+1}: Applying LPF at {lpf_cutoff} Hz (order {lpf_order})")
                signal = pf.dsp.filter.butterworth(
                    signal, 
                    N=lpf_order, 
                    frequency=lpf_cutoff, 
                    btype='lowpass'
                )
            
            # Ensure it's a 1D array and append
            channel_data = np.asarray(signal.time).flatten()
            filtered_channels.append(channel_data)
        
        # Combine channels back to stereo (samples x channels)
        filtered_data = np.stack(filtered_channels, axis=-1)
    else:
        # Process mono
        print(f"  Processing mono (1 channel)...")
        signal = pf.Signal(audio_data, fs)
        
        # Apply high-pass filter
        if apply_hpf:
            print(f"    Applying HPF at {hpf_cutoff} Hz (order {hpf_order})")
            signal = pf.dsp.filter.butterworth(
                signal, 
                N=hpf_order, 
                frequency=hpf_cutoff, 
                btype='highpass'
            )
        
        # Apply low-pass filter
        if apply_lpf:
            print(f"    Applying LPF at {lpf_cutoff} Hz (order {lpf_order})")
            signal = pf.dsp.filter.butterworth(
                signal, 
                N=lpf_order, 
                frequency=lpf_cutoff, 
                btype='lowpass'
            )
        
        filtered_data = np.asarray(signal.time).flatten()
    
    # Analysis after filtering
    max_after = np.max(np.abs(filtered_data))
    rms_after = np.sqrt(np.mean(filtered_data**2))
    
    print(f"\nAfter filtering:")
    print(f"  Peak level: {max_after:.6f} ({20*np.log10(max_after + 1e-10):.1f} dBFS)")
    print(f"  RMS level:  {rms_after:.6f} ({20*np.log10(rms_after + 1e-10):.1f} dBFS)")
    
    # Show RMS change
    rms_change_db = 20 * np.log10(rms_after / (rms_before + 1e-10))
    print(f"  RMS change: {rms_change_db:+.1f} dB")
    
    # Check for clipping
    if max_after > 0.99:
        print(f"\n  ? WARNING: Signal is clipping! Normalizing...")
        filtered_data = filtered_data * (0.99 / max_after)
        max_after = 0.99
    
    # Convert back to int16 for saving
    if bit_depth == 16:
        filtered_data_int = np.clip(filtered_data * 32767, -32768, 32767).astype(np.int16)
    else:
        filtered_data_int = np.clip(filtered_data * 2147483647, -2147483648, 2147483647).astype(np.int32)
    
    # Ensure data is C-contiguous
    filtered_data_int = np.ascontiguousarray(filtered_data_int)
    
    # Save filtered file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving filtered audio...")
    print(f"  Output shape: {filtered_data_int.shape}")
    print(f"  Output dtype: {filtered_data_int.dtype}")
    
    wavfile.write(output_file, int(fs), filtered_data_int)
    
    print(f"\n? Filtered audio saved to: {output_file}")
    print(f"  Format: WAV")
    print(f"  Bit depth: {bit_depth}")
    print("="*70)
    
    return filtered_data, fs

def main():
    """Main function"""
    # Check if input file exists
    input_path = Path(INPUT_WAV)
    if not input_path.exists():
        print(f"Error: Input file not found: {INPUT_WAV}")
        return
    
    # Check if at least one filter is enabled
    if not APPLY_HPF and not APPLY_LPF:
        print("\n? WARNING: Both filters are disabled!")
        print("Enable at least one filter (APPLY_HPF or APPLY_LPF) in the configuration.")
        return
    
    # Apply filters
    filtered_audio, fs = apply_butterworth_filters(
        input_file=INPUT_WAV,
        output_file=OUTPUT_WAV,
        apply_hpf=APPLY_HPF,
        hpf_cutoff=HPF_CUTOFF,
        hpf_order=HPF_ORDER,
        apply_lpf=APPLY_LPF,
        lpf_cutoff=LPF_CUTOFF,
        lpf_order=LPF_ORDER
    )
    
    if filtered_audio is not None:
        print("\n? Processing complete!")

if __name__ == "__main__":
    main()