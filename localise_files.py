#localises audio files in a folder using Berta 3D spatial audio engine. works 08/12/25

#!/usr/bin/env python3
"""
Localize audio files using BeRTA 3D spatial audio engine
Creates binaural versions of audio files positioned at specific azimuth/distance
Records output from BeRTA using VB-Audio Virtual Cable
"""

import os
import time
import math
import soundfile as sf
import numpy as np
import sounddevice as sd
from pathlib import Path
import threading

# BeRTA OSC integration using pythonosc
try:
    from pythonosc.udp_client import SimpleUDPClient
    BERTA_AVAILABLE = True
    print("BeRTA OSC integration available using pythonosc")
except ImportError:
    BERTA_AVAILABLE = False
    print("Error: pythonosc not available - install with: pip install python-osc")
    exit(1)


class BertaAudioLocalizer:
    def __init__(self, input_folder, output_folder, berta_ip='127.0.0.1', berta_port=10017):
        """
        Initialize BeRTA audio localizer
        
        Args:
            input_folder: Folder containing source audio files
            output_folder: Folder to save localized audio files
            berta_ip: BeRTA OSC server IP
            berta_port: BeRTA OSC server port
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.berta_ip = berta_ip
        self.berta_port = berta_port
        self.osc_client = None
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize BeRTA connection
        self._init_berta()
    
    def _init_berta(self):
        """Initialize connection to BeRTA"""
        try:
            self.osc_client = SimpleUDPClient(self.berta_ip, self.berta_port)
            print(f"BeRTA OSC client connected: {self.berta_ip}:{self.berta_port}")
            
            # Configure BeRTA for audio processing
            self.osc_client.send_message("/listener/enableSpatialization", ["listener0", True])
            print("✓ Enabled spatialization")
            
            self.osc_client.send_message("/listener/enableDistanceAttenuation", ["DirectPath", True])
            print("✓ Enabled distance attenuation")
            
            self.osc_client.send_message("/listener/setDistanceAttenuationFactor", ["DirectPath", -6.0])
            print("✓ Set distance attenuation to -6dB per doubling")
            
            self.osc_client.send_message("/listener/enableITD", ["listener0", True])
            print("✓ Enabled ITD (Interaural Time Difference)")
            
            self.osc_client.send_message("/listener/enableNearFieldEffect", ["listener0", True])
            print("✓ Enabled near field effects")
            
            self.osc_client.send_message("/listener/enableInterpolation", ["listener0", True])
            print("✓ Enabled HRTF interpolation")
            
        except Exception as e:
            print(f"Error initializing BeRTA: {e}")
            raise
    
    def localize_file(self, audio_file, azimuth_deg, distance_m, duration_sec=10, 
                     record_device=1, source_id="source1"):
        """
        Localize a single audio file using BeRTA
        
        Args:
            audio_file: Path to input audio file
            azimuth_deg: Azimuth angle in degrees (-180 to 180, 0=front, 90=right, -90=left)
            distance_m: Distance in meters (realistic: 0.5 to 100m)
            duration_sec: Duration to record localized audio
            record_device: Audio device index for recording (e.g., VB-Audio input)
            source_id: BeRTA source identifier
            
        Returns:
            output_file_path or None if error
        """
        try:
            # Load audio file to check it exists
            print(f"\nLoading: {audio_file.name}")
            audio_data, fs = sf.read(str(audio_file))
            original_duration = len(audio_data) / fs
            print(f"  Sample rate: {fs} Hz, Duration: {original_duration:.2f}s")
            
            # Resample to 48kHz if needed
            if fs != 48000:
                from scipy import signal as sp_signal
                num_samples = int(len(audio_data) * 48000 / fs)
                if audio_data.ndim == 1:
                    audio_data = sp_signal.resample(audio_data, num_samples)
                else:
                    # Resample each channel separately for stereo
                    audio_data = np.column_stack([
                        sp_signal.resample(audio_data[:, ch], num_samples)
                        for ch in range(audio_data.shape[1])
                    ])
                fs = 48000
                print(f"  Resampled to 48kHz")
                
                # Save resampled version for BeRTA to load
                temp_file = self.output_folder / f"_temp_resampled_{audio_file.name}"
                sf.write(str(temp_file), audio_data, fs)
                audio_file_path = str(temp_file.absolute())
                print(f"  Saved temp 48kHz version for BeRTA")
            else:
                audio_file_path = str(audio_file.absolute())
            
            # Configure BeRTA source position
            print(f"Setting up BeRTA source at azimuth={azimuth_deg}°, distance={distance_m}m")
            
            # Convert spherical (azimuth, distance) to Cartesian (x, y, z)
            azimuth_rad = math.radians(azimuth_deg)
            x = distance_m * math.cos(azimuth_rad)
            y = -distance_m * math.sin(azimuth_rad)
            z = 0.0  # Ear level
            
            # Load audio file into BeRTA source using /source/loadSource
            print(f"  Loading into BeRTA: {audio_file_path}")
            self.osc_client.send_message("/source/loadSource", [
                source_id,
                audio_file_path,
                "OmnidirectionalModel"
            ])
            time.sleep(0.5)  # Give BeRTA time to load
            
            # Set source location
            self.osc_client.send_message("/source/location", [source_id, x, y, z])
            print(f"  Position set to ({x:.2f}, {y:.2f}, {z:.2f}) metres")
            
            # Reset listener to neutral position
            self.osc_client.send_message("/listener/orientation", ["listener0", 0.0, 0.0, 0.0])
            self.osc_client.send_message("/listener/location", ["listener0", 0.0, 0.0, 0.0])
            
            recorded_audio = None
            
            if record_device is not None:
                print(f"  Starting playback and recording ({duration_sec}s)...")
                
                recorded_frames = []
                
                def record_thread_func():
                    """Record from VB-Audio while playback happens"""
                    try:
                        # Create recording stream at 48kHz
                        stream = sd.InputStream(device=record_device, channels=2, 
                                              samplerate=48000, blocksize=2048)
                        stream.start()
                        
                        # Read for the specified duration
                        num_samples = int(duration_sec * 48000)
                        recorded_frames.append(stream.read(num_samples)[0])
                        
                        stream.stop()
                        stream.close()
                    except Exception as e:
                        print(f"    Recording error: {e}")
                
                # Start recording thread
                rec_thread = threading.Thread(target=record_thread_func)
                rec_thread.daemon = True
                rec_thread.start()
                
                # Give recording thread time to initialize
                time.sleep(0.3)
            
            # Start BeRTA playback
            print(f"  Playing audio...")
            self.osc_client.send_message("/source/play", [source_id])
            
            # Small delay to ensure BeRTA starts outputting before we check recording
            time.sleep(0.1)
            
            # Wait for duration
            time.sleep(duration_sec)
            
            # Stop playback
            self.osc_client.send_message("/source/stop", [source_id])
            
            # Clean up source for next file
            time.sleep(0.1)
            self.osc_client.send_message("/source/removeSource", [source_id])
            
            # Wait for recording to finish
            if record_device is not None:
                rec_thread.join(timeout=duration_sec + 2)
                
                if recorded_frames:
                    recorded_audio = recorded_frames[0]
                    print(f"  Recording complete ({len(recorded_audio)} samples)")
                    
                    # Check audio amplitude
                    max_amp = np.max(np.abs(recorded_audio))
                    print(f"  Max amplitude: {max_amp:.6f}")
                    if max_amp < 0.001:
                        print(f"  ⚠ WARNING: Recording is nearly silent!")
                        print(f"     Check BeRTA output routing to VB-Audio Virtual Cable")
            
            # Create output filename
            stem = audio_file.stem
            output_filename = f"{stem}_az{azimuth_deg:+.0f}_dist{distance_m:.1f}m.wav"
            output_path = self.output_folder / output_filename
            
            # Save recorded audio
            if recorded_audio is not None and len(recorded_audio) > 0:
                try:
                    sf.write(str(output_path), recorded_audio, 48000)
                    print(f"  ✓ Saved: {output_filename}")
                    
                    # Clean up temp resampled file if it exists
                    if 'temp_file' in locals():
                        try:
                            temp_file.unlink()
                            print(f"  Cleaned up temp file")
                        except:
                            pass
                    
                    return output_path
                except Exception as e:
                    print(f"  Error saving audio: {e}")
                    return None
            elif record_device is None:
                print(f"  (Audio played through BeRTA - no recording device specified)")
                return output_path
            else:
                print(f"  Warning: No audio recorded")
                return None
            
        except Exception as e:
            print(f"Error localizing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def localize_folder(self, azimuth_deg=-45, distance_m=10, duration_sec=10, record_device=1):
        """
        Localize all audio files in the input folder
        
        Args:
            azimuth_deg: Azimuth angle in degrees
            distance_m: Distance in meters
            duration_sec: Duration to record each file
            record_device: Audio device index for recording (e.g., VB-Audio input)
        """
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
        audio_files = [f for f in self.input_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"No audio files found in {self.input_folder}")
            return
        
        print(f"\nFound {len(audio_files)} audio files")
        print(f"Localizing to: azimuth={azimuth_deg}°, distance={distance_m}m")
        if record_device is not None:
            print(f"Recording from device: {record_device}")
        print("=" * 70)
        
        output_files = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            output_file = self.localize_file(audio_file, azimuth_deg, distance_m, duration_sec, 
                                           record_device=record_device)
            if output_file:
                output_files.append(output_file)
        
        print("\n" + "=" * 70)
        print(f"Localization complete!")
        print(f"Output folder: {self.output_folder}")
        print(f"Successfully processed: {len(output_files)} files")
        
        return output_files
    
    def close(self):
        """Close BeRTA connection"""
        if self.osc_client:
            try:
                self.osc_client.send_message("/source/stop", ["source1"])
            except:
                pass
        print("\nBeRTA connection closed")


def main():
    """Main function to localize all audio files in loudspeaker_stimuli folder"""
    
    # file locations
    script_dir = Path(__file__).resolve().parent
    input_folder = script_dir / "loudspeaker_stimuli"
    output_folder = script_dir / "localised_stimuli"
    
    # Localization parameters
    azimuth = -45.0  # degrees
    distance = 1.0   # meters
    duration = 5    # seconds
    record_device = 1  # VB-Audio Virtual Cable
    
    print("BeRTA Audio Localizer")
    print("=" * 70)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Target position: azimuth={azimuth}°, distance={distance}m")
    print(f"Recording device: {record_device} (VB-Audio Virtual Cable)")
    print()
    
    # Create localizer and process files
    localizer = BertaAudioLocalizer(input_folder, output_folder)
    
    try:
        localizer.localize_folder(azimuth_deg=azimuth, distance_m=distance, 
                                 duration_sec=duration, record_device=record_device)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        localizer.close()


if __name__ == "__main__":
    main()
