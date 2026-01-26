
# Play a stimulus with simultaneous masker
# Supports two modes:
#   1. Stimulus via HEADPHONES + Masker via HEADPHONES
#   2. Stimulus via LOUDSPEAKER + Masker via HEADPHONES

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
from scipy import signal as scipy_signal
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

# Playback mode: 'headphones' or 'loudspeaker'
PLAYBACK_MODE = 'headphones'  # 'headphones' = both via headphones, 'loudspeaker' = stimulus via speaker

# Stimulus file (target signal)
STIMULUS_FILE = r'C:\Users\tim_e\source\repos\auditory_distance\binaural_recordings\ISTS-V1.0_60s_24bit_2_binaural_recording.wav'

# Masker file (noise/interference)
MASKER_FILE = r'C:\Users\tim_e\source\repos\auditory_distance\original_audios\noise\brown_noise_5s.wav'

# Audio device indices (use sd.query_devices() to find correct indices)
HEADPHONE_DEVICE = 5    # Device for headphone output
LOUDSPEAKER_DEVICE = 6  # Device for loudspeaker output

# Level adjustments (in dB)
STIMULUS_GAIN_DB = 0.0   # Gain for stimulus signal
MASKER_GAIN_DB = 0.0     # Gain for masker signal

# Signal-to-Noise Ratio (SNR) in dB - alternative to separate gains
USE_SNR = False          # If True, uses SNR instead of separate gains
TARGET_SNR_DB = 0.0      # Desired SNR (stimulus level relative to masker)

# Timing
PRE_STIMULUS_DELAY = 0.5   # Seconds of masker before stimulus starts
POST_STIMULUS_DELAY = 0.5  # Seconds of masker after stimulus ends

# Resampling
AUTO_RESAMPLE = True     # Automatically resample to match sample rates

# ============================================================================

def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio array (mono or stereo)
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    print(f"  Resampling from {orig_sr} Hz to {target_sr} Hz...")
    
    # Calculate number of samples in resampled signal
    num_samples = int(len(audio) * target_sr / orig_sr)
    
    if audio.ndim == 1:
        # Mono
        resampled = scipy_signal.resample(audio, num_samples)
    else:
        # Stereo - resample each channel
        resampled = np.zeros((num_samples, audio.shape[1]))
        for ch in range(audio.shape[1]):
            resampled[:, ch] = scipy_signal.resample(audio[:, ch], num_samples)
    
    return resampled

def load_and_prepare_audio(stimulus_file, masker_file, stimulus_gain_db, masker_gain_db,
                           use_snr=False, target_snr_db=0.0, auto_resample=True):
    """
    Load stimulus and masker files and prepare them for playback.
    
    Args:
        stimulus_file: Path to stimulus WAV file
        masker_file: Path to masker WAV file
        stimulus_gain_db: Gain adjustment for stimulus in dB
        masker_gain_db: Gain adjustment for masker in dB
        use_snr: If True, adjust levels based on SNR instead
        target_snr_db: Target SNR in dB
        auto_resample: If True, automatically resample to match sample rates
        
    Returns:
        stimulus_audio, masker_audio, sample_rate
    """
    print("\n" + "="*70)
    print("LOADING AUDIO FILES")
    print("="*70)
    
    # Load stimulus
    print(f"\nStimulus: {Path(stimulus_file).name}")
    stimulus_audio, stim_fs = sf.read(stimulus_file)
    print(f"  Sample rate: {stim_fs} Hz")
    print(f"  Duration: {len(stimulus_audio) / stim_fs:.2f} seconds")
    print(f"  Channels: {stimulus_audio.shape[1] if stimulus_audio.ndim > 1 else 1}")
    
    # Load masker
    print(f"\nMasker: {Path(masker_file).name}")
    masker_audio, mask_fs = sf.read(masker_file)
    print(f"  Sample rate: {mask_fs} Hz")
    print(f"  Duration: {len(masker_audio) / mask_fs:.2f} seconds")
    print(f"  Channels: {masker_audio.shape[1] if masker_audio.ndim > 1 else 1}")
    
    # Check sample rates match
    if stim_fs != mask_fs:
        if auto_resample:
            print(f"\nSample rate mismatch detected!")
            print(f"  Stimulus: {stim_fs} Hz")
            print(f"  Masker: {mask_fs} Hz")
            
            # Resample masker to match stimulus (usually better to keep stimulus untouched)
            masker_audio = resample_audio(masker_audio, mask_fs, stim_fs)
            mask_fs = stim_fs
            
            print(f"  New masker duration: {len(masker_audio) / mask_fs:.2f} seconds")
        else:
            raise ValueError(f"Sample rate mismatch: Stimulus={stim_fs} Hz, Masker={mask_fs} Hz")
    
    fs = stim_fs
    
    # Calculate original RMS levels
    stim_rms = np.sqrt(np.mean(stimulus_audio**2))
    mask_rms = np.sqrt(np.mean(masker_audio**2))
    
    print("\n" + "-"*70)
    print("ORIGINAL LEVELS")
    print("-"*70)
    print(f"  Stimulus RMS: {stim_rms:.6f} ({20*np.log10(stim_rms + 1e-10):.1f} dBFS)")
    print(f"  Masker RMS:   {mask_rms:.6f} ({20*np.log10(mask_rms + 1e-10):.1f} dBFS)")
    
    # Apply gains or adjust for SNR
    if use_snr:
        # Calculate gains to achieve target SNR
        # SNR = 20*log10(stim_rms / mask_rms)
        # Target: stim_rms_new / mask_rms_new = 10^(SNR/20)
        
        print(f"\nAdjusting for target SNR: {target_snr_db:.1f} dB")
        
        # Keep masker at original level, adjust stimulus
        masker_gain_linear = 1.0
        stimulus_gain_linear = (10 ** (target_snr_db / 20)) * (mask_rms / stim_rms)
        
        print(f"  Masker gain: 0.0 dB (linear: 1.0)")
        print(f"  Stimulus gain: {20*np.log10(stimulus_gain_linear):.1f} dB (linear: {stimulus_gain_linear:.3f})")
    else:
        # Use specified gains
        stimulus_gain_linear = 10 ** (stimulus_gain_db / 20)
        masker_gain_linear = 10 ** (masker_gain_db / 20)
        
        print(f"\nApplying gains:")
        print(f"  Stimulus: {stimulus_gain_db:+.1f} dB (linear: {stimulus_gain_linear:.3f})")
        print(f"  Masker:   {masker_gain_db:+.1f} dB (linear: {masker_gain_linear:.3f})")
    
    # Apply gains
    stimulus_audio = stimulus_audio * stimulus_gain_linear
    masker_audio = masker_audio * masker_gain_linear
    
    # Calculate new RMS levels
    stim_rms_new = np.sqrt(np.mean(stimulus_audio**2))
    mask_rms_new = np.sqrt(np.mean(masker_audio**2))
    
    # Calculate achieved SNR
    achieved_snr = 20 * np.log10(stim_rms_new / (mask_rms_new + 1e-10))
    
    print("\n" + "-"*70)
    print("ADJUSTED LEVELS")
    print("-"*70)
    print(f"  Stimulus RMS: {stim_rms_new:.6f} ({20*np.log10(stim_rms_new + 1e-10):.1f} dBFS)")
    print(f"  Masker RMS:   {mask_rms_new:.6f} ({20*np.log10(mask_rms_new + 1e-10):.1f} dBFS)")
    print(f"  Achieved SNR: {achieved_snr:.1f} dB")
    
    # Check for clipping
    stim_peak = np.max(np.abs(stimulus_audio))
    mask_peak = np.max(np.abs(masker_audio))
    
    if stim_peak > 0.99:
        print(f"\n  WARNING: Stimulus will clip (peak: {stim_peak:.3f})!")
    if mask_peak > 0.99:
        print(f"  WARNING: Masker will clip (peak: {mask_peak:.3f})!")
    
    return stimulus_audio, masker_audio, fs

def play_headphones_mode(stimulus_audio, masker_audio, fs, device, 
                         pre_delay, post_delay):
    """
    Play both stimulus and masker through headphones.
    
    Args:
        stimulus_audio: Stimulus audio array
        masker_audio: Masker audio array
        fs: Sample rate
        device: Headphone device index
        pre_delay: Seconds before stimulus starts
        post_delay: Seconds after stimulus ends
    """
    print("\n" + "="*70)
    print("PLAYBACK MODE: HEADPHONES")
    print("="*70)
    print(f"\nBoth stimulus and masker will play through headphones (device {device})")
    
    # Ensure both are stereo for headphones
    if stimulus_audio.ndim == 1:
        stimulus_audio = np.column_stack([stimulus_audio, stimulus_audio])
    if masker_audio.ndim == 1:
        masker_audio = np.column_stack([masker_audio, masker_audio])
    
    # Calculate lengths
    stim_samples = len(stimulus_audio)
    mask_samples = len(masker_audio)
    
    pre_samples = int(pre_delay * fs)
    post_samples = int(post_delay * fs)
    
    total_samples = pre_samples + stim_samples + post_samples
    
    # Check if masker is long enough
    if mask_samples < total_samples:
        print(f"\nWARNING: Masker too short! Looping masker...")
        # Loop masker to required length
        repeats = int(np.ceil(total_samples / mask_samples))
        masker_audio = np.tile(masker_audio, (repeats, 1))[:total_samples]
    else:
        # Trim masker to required length
        masker_audio = masker_audio[:total_samples]
    
    # Create combined audio
    combined = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Add masker for entire duration
    combined[:] = masker_audio[:total_samples]
    
    # Add stimulus with delay
    combined[pre_samples:pre_samples + stim_samples] += stimulus_audio
    
    # Check for clipping in combined signal
    peak = np.max(np.abs(combined))
    if peak > 0.99:
        print(f"\nWARNING: Combined signal will clip (peak: {peak:.3f})!")
        print("Normalizing to prevent clipping...")
        combined = combined * (0.99 / peak)
    
    # Play
    print(f"\nPlaying audio...")
    print(f"  Total duration: {total_samples / fs:.2f} seconds")
    print(f"  Pre-stimulus masker: {pre_delay:.2f} s")
    print(f"  Stimulus duration: {stim_samples / fs:.2f} s")
    print(f"  Post-stimulus masker: {post_delay:.2f} s")
    print("\nStarting playback in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("\nPLAYING...")
    sd.play(combined, samplerate=fs, device=device)
    sd.wait()
    print("PLAYBACK COMPLETE")

def play_loudspeaker_mode(stimulus_audio, masker_audio, fs, 
                          loudspeaker_device, headphone_device,
                          pre_delay, post_delay):
    """
    Play stimulus through loudspeaker and masker through headphones simultaneously.
    
    Args:
        stimulus_audio: Stimulus audio array
        masker_audio: Masker audio array
        fs: Sample rate
        loudspeaker_device: Loudspeaker device index
        headphone_device: Headphone device index
        pre_delay: Seconds before stimulus starts
        post_delay: Seconds after stimulus ends
    """
    print("\n" + "="*70)
    print("PLAYBACK MODE: LOUDSPEAKER + HEADPHONES")
    print("="*70)
    print(f"\nStimulus: Loudspeaker (device {loudspeaker_device})")
    print(f"Masker:   Headphones (device {headphone_device})")
    
    # Convert stimulus to mono for loudspeaker if stereo
    if stimulus_audio.ndim > 1:
        stimulus_mono = np.mean(stimulus_audio, axis=1)
    else:
        stimulus_mono = stimulus_audio
    
    # Ensure masker is stereo for headphones
    if masker_audio.ndim == 1:
        masker_audio = np.column_stack([masker_audio, masker_audio])
    
    # Calculate lengths
    stim_samples = len(stimulus_mono)
    mask_samples = len(masker_audio)
    
    pre_samples = int(pre_delay * fs)
    post_samples = int(post_delay * fs)
    
    total_samples = pre_samples + stim_samples + post_samples
    
    # Check if masker is long enough
    if mask_samples < total_samples:
        print(f"\nWARNING: Masker too short! Looping masker...")
        repeats = int(np.ceil(total_samples / mask_samples))
        masker_audio = np.tile(masker_audio, (repeats, 1))[:total_samples]
    else:
        masker_audio = masker_audio[:total_samples]
    
    # Create stimulus with silence padding
    stimulus_padded = np.zeros(total_samples, dtype=np.float32)
    stimulus_padded[pre_samples:pre_samples + stim_samples] = stimulus_mono
    
    # Play using threading to handle different devices
    print(f"\nPlaying audio...")
    print(f"  Total duration: {total_samples / fs:.2f} seconds")
    print(f"  Pre-stimulus masker: {pre_delay:.2f} s")
    print(f"  Stimulus duration: {stim_samples / fs:.2f} s")
    print(f"  Post-stimulus masker: {post_delay:.2f} s")
    print("\nStarting playback in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("\nPLAYING...")
    
    # Use OutputStream for better control
    # Create output streams
    masker_stream = sd.OutputStream(
        samplerate=fs,
        device=headphone_device,
        channels=2,
        dtype='float32'
    )
    
    stimulus_stream = sd.OutputStream(
        samplerate=fs,
        device=loudspeaker_device,
        channels=1,
        dtype='float32'
    )
    
    # Start both streams
    masker_stream.start()
    stimulus_stream.start()
    
    print(f"  Masker stream started on device {headphone_device}")
    print(f"  Stimulus stream started on device {loudspeaker_device}")
    
    # Write audio data in chunks
    chunk_size = 1024
    total_chunks = int(np.ceil(total_samples / chunk_size))
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        
        # Write masker chunk to headphones
        masker_chunk = masker_audio[start_idx:end_idx]
        masker_stream.write(masker_chunk.astype(np.float32))
        
        # Write stimulus chunk to loudspeaker
        stimulus_chunk = stimulus_padded[start_idx:end_idx]
        stimulus_stream.write(stimulus_chunk.astype(np.float32))
    
    # Stop and close streams
    masker_stream.stop()
    masker_stream.close()
    stimulus_stream.stop()
    stimulus_stream.close()
    
    print("PLAYBACK COMPLETE")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("STIMULUS + MASKER PLAYBACK")
    print("="*70)
    
    # Check files exist
    if not Path(STIMULUS_FILE).exists():
        print(f"\nERROR: Stimulus file not found: {STIMULUS_FILE}")
        return
    
    if not Path(MASKER_FILE).exists():
        print(f"\nERROR: Masker file not found: {MASKER_FILE}")
        return
    
    # Display available audio devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    # Load and prepare audio
    stimulus_audio, masker_audio, fs = load_and_prepare_audio(
        STIMULUS_FILE, MASKER_FILE,
        STIMULUS_GAIN_DB, MASKER_GAIN_DB,
        USE_SNR, TARGET_SNR_DB,
        AUTO_RESAMPLE
    )
    
    # Play based on selected mode
    if PLAYBACK_MODE.lower() == 'headphones':
        play_headphones_mode(
            stimulus_audio, masker_audio, fs,
            HEADPHONE_DEVICE,
            PRE_STIMULUS_DELAY, POST_STIMULUS_DELAY
        )
    elif PLAYBACK_MODE.lower() == 'loudspeaker':
        play_loudspeaker_mode(
            stimulus_audio, masker_audio, fs,
            LOUDSPEAKER_DEVICE, HEADPHONE_DEVICE,
            PRE_STIMULUS_DELAY, POST_STIMULUS_DELAY
        )
    else:
        print(f"\nERROR: Invalid PLAYBACK_MODE: {PLAYBACK_MODE}")
        print("Must be 'headphones' or 'loudspeaker'")
        return
    
    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
