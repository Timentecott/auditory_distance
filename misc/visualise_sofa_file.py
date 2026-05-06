# -*- coding: utf-8 -*-
# Visualize SOFA (Spatially Oriented Format for Acoustics) files
# Displays metadata, impulse responses, magnitude/phase, and spatial information

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sofar as sf
import pyfar as pf

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input SOFA file
SOFA_FILE = r'C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\360-BRIR-FOAIR-database\Binaural\SOFA\C4m.sofa' #a02 is 1m from the sound source

# Visualization settings
SHOW_METADATA = True        # Display SOFA file metadata
SHOW_POSITIONS = True       # Plot source positions
SHOW_IMPULSE_RESPONSES = True  # Plot time-domain impulse responses
SHOW_MAGNITUDE = True       # Plot magnitude responses
SHOW_PHASE = False          # Plot phase responses
SHOW_SPECTROGRAM = False    # Plot spectrograms

# Which measurements to plot (indices, or 'all' for all, or 'first_10' for first 10)
MEASUREMENTS_TO_PLOT = 'first_10'  # Can be 'all', 'first_10', or list like [0, 5, 10]

# Ear selection for plots
PLOT_LEFT_EAR = True
PLOT_RIGHT_EAR = True

# Frequency range for magnitude plots
FREQ_RANGE = [20, 20000]  # Hz

# Skip SOFA verification (useful for files with custom entries)
SKIP_VERIFICATION = True

# ============================================================================

def load_sofa_file(sofa_path, skip_verification=False):
    """
    Load and parse a SOFA file.
    
    Args:
        sofa_path: Path to SOFA file
        skip_verification: If True, skips SOFA format verification
        
    Returns:
        sofa: Loaded SOFA object
    """
    print("\n" + "="*70)
    print("LOADING SOFA FILE")
    print("="*70)
    
    print(f"\nFile: {Path(sofa_path).name}")
    print(f"Path: {sofa_path}")
    
    try:
        # First try with verification
        if not skip_verification:
            print("\nAttempting to load with verification...")
            sofa = sf.read_sofa(sofa_path)
        else:
            print("\nLoading with verification disabled...")
            sofa = sf.read_sofa(sofa_path, verify=False)
        
        print("CHECK SOFA file loaded successfully")
        
        # Try to verify and show any issues
        if skip_verification:
            print("\nChecking SOFA format compliance...")
            try:
                sofa.verify()
                print("  File is fully compliant")
            except Exception as verify_error:
                print(f"  WARNING: Verification issues found (file still usable):")
                print(f"  {verify_error}")
        
        return sofa
        
    except Exception as e:
        # If verification failed, try without verification
        if not skip_verification:
            print(f"\nInitial load failed: {e}")
            print("Retrying with verification disabled...")
            try:
                sofa = sf.read_sofa(sofa_path, verify=False)
                print("CHECK SOFA file loaded successfully (verification skipped)")
                return sofa
            except Exception as e2:
                print(f"X Error loading SOFA file: {e2}")
                return None
        else:
            print(f"X Error loading SOFA file: {e}")
            return None

def display_metadata(sofa):
    """Display SOFA file metadata and information."""
    print("\n" + "="*70)
    print("SOFA FILE METADATA")
    print("="*70)
    
    # Convention and version
    print(f"\nConvention: {sofa.GLOBAL_SOFAConventions}")
    print(f"Version: {sofa.GLOBAL_SOFAConventionsVersion}")
    print(f"Data Type: {sofa.GLOBAL_DataType}")
    
    # Room and listener info
    if hasattr(sofa, 'GLOBAL_RoomType'):
        print(f"Room Type: {sofa.GLOBAL_RoomType}")
    if hasattr(sofa, 'GLOBAL_ListenerShortName'):
        print(f"Listener: {sofa.GLOBAL_ListenerShortName}")
    
    # Custom attributes that might exist
    if hasattr(sofa, 'GLOBAL_RoomShortName'):
        print(f"Room Name: {sofa.GLOBAL_RoomShortName}")
    if hasattr(sofa, 'GLOBAL_RoomVolume'):
        print(f"Room Volume: {sofa.GLOBAL_RoomVolume}")
    if hasattr(sofa, 'GLOBAL_ListenerDescription'):
        print(f"Listener Description: {sofa.GLOBAL_ListenerDescription}")
    
    # Data dimensions - get shape from actual data arrays
    ir_shape = sofa.Data_IR.shape
    
    print(f"\nData dimensions:")
    print(f"  Measurements (M): {ir_shape[0]}")
    print(f"  Receivers (R): {ir_shape[1]}")
    print(f"  Samples (N): {ir_shape[2]}")
    
    # Sampling rate
    print(f"  Sampling rate: {sofa.Data_SamplingRate} Hz")
    
    # Source positions
    print(f"\nSource positions:")
    print(f"  Type: {sofa.SourcePosition_Type}")
    print(f"  Units: {sofa.SourcePosition_Units}")
    print(f"  Number of positions: {len(sofa.SourcePosition)}")
    
    # Check if single position for multiple measurements
    if len(sofa.SourcePosition) == 1 and ir_shape[0] > 1:
        print(f"  NOTE: Single source position for {ir_shape[0]} measurements")
        print(f"        (BRIR dataset - fixed source, rotating listener)")
    
    # Listener View/Position - this is what varies!
    if hasattr(sofa, 'ListenerView'):
        print(f"\nListener View (head orientation):")
        print(f"  Type: {sofa.ListenerView_Type}")
        print(f"  Units: {sofa.ListenerView_Units}")
        print(f"  Number of views: {len(sofa.ListenerView)}")
        
        if len(sofa.ListenerView) == ir_shape[0]:
            # Calculate azimuth differences
            azimuths = sofa.ListenerView[:, 0]
            if len(azimuths) > 1:
                azimuth_diffs = np.diff(azimuths)
                mean_diff = np.mean(azimuth_diffs)
                print(f"  Azimuth range: {azimuths.min():.1f} to {azimuths.max():.1f} degrees")
                print(f"  Mean azimuth step: {mean_diff:.1f} degrees")
                print(f"  CHECK Listener rotates {mean_diff:.1f} deg between measurements")
    
    if hasattr(sofa, 'ListenerPosition'):
        print(f"\nListener Position:")
        print(f"  Type: {sofa.ListenerPosition_Type}")
        print(f"  Units: {sofa.ListenerPosition_Units}")
        print(f"  Number of positions: {len(sofa.ListenerPosition)}")
    
    # Receiver positions
    print(f"\nReceiver positions:")
    print(f"  Type: {sofa.ReceiverPosition_Type}")
    print(f"  Units: {sofa.ReceiverPosition_Units}")
    
    # Additional attributes
    if hasattr(sofa, 'GLOBAL_Comment'):
        print(f"\nComment: {sofa.GLOBAL_Comment}")
    if hasattr(sofa, 'GLOBAL_History'):
        print(f"History: {sofa.GLOBAL_History}")
    if hasattr(sofa, 'GLOBAL_DatabaseName'):
        print(f"Database: {sofa.GLOBAL_DatabaseName}")

def plot_source_positions(sofa):
    """Plot source positions and listener views."""
    print("\n" + "="*70)
    print("PLOTTING POSITIONS")
    print("="*70)
    
    # Determine what to plot
    has_listener_view = hasattr(sofa, 'ListenerView') and len(sofa.ListenerView) > 1
    has_multiple_sources = len(sofa.SourcePosition) > 1
    
    if has_listener_view:
        # Plot listener views (rotating head)
        plot_listener_views(sofa)
    
    if has_multiple_sources:
        # Plot source positions
        plot_source_positions_data(sofa)
    elif not has_listener_view:
        # Just show the single source
        plot_source_positions_data(sofa)

def plot_listener_views(sofa):
    """Plot listener head orientations over measurements."""
    print("\n  Plotting listener head orientations...")
    
    views = sofa.ListenerView
    view_type = sofa.ListenerView_Type
    
    fig = plt.figure(figsize=(15, 5))
    
    if view_type == 'spherical':
        # Spherical coordinates: azimuth, elevation, roll (usually)
        azimuth = views[:, 0]
        elevation = views[:, 1]
        
        # Plot 1: Azimuth over measurements
        ax1 = fig.add_subplot(131)
        ax1.plot(np.arange(len(azimuth)), azimuth, 'b.-', markersize=4)
        ax1.set_xlabel('Measurement Index')
        ax1.set_ylabel('Azimuth (deg)')
        ax1.set_title('Listener Head Azimuth')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Polar plot of azimuth
        ax2 = fig.add_subplot(132, projection='polar')
        scatter = ax2.scatter(np.deg2rad(azimuth), np.ones_like(azimuth), 
                             c=np.arange(len(azimuth)), cmap='viridis', s=50)
        ax2.set_title('Listener Head Orientations\n(Top View)')
        plt.colorbar(scatter, ax=ax2, label='Measurement')
        
        # Plot 3: 2D scatter with measurement number
        ax3 = fig.add_subplot(133)
        scatter = ax3.scatter(azimuth, elevation, c=np.arange(len(azimuth)), 
                             cmap='viridis', s=50)
        ax3.set_xlabel('Azimuth (deg)')
        ax3.set_ylabel('Elevation (deg)')
        ax3.set_title('Listener Head Orientation\n(Azimuth vs Elevation)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Measurement')
        
        print(f"    Azimuth range: {azimuth.min():.1f} to {azimuth.max():.1f} deg")
        print(f"    Elevation range: {elevation.min():.1f} to {elevation.max():.1f} deg")
        
        if len(azimuth) > 1:
            azimuth_diffs = np.diff(azimuth)
            print(f"    Azimuth steps: mean={np.mean(azimuth_diffs):.1f} deg, "
                  f"min={np.min(azimuth_diffs):.1f} deg, max={np.max(azimuth_diffs):.1f} deg")
    
    plt.tight_layout()
    print("    CHECK Listener view plot created")

def plot_source_positions_data(sofa):
    """Plot source positions in 3D or 2D."""
    print("\n  Plotting source positions...")
    
    positions = sofa.SourcePosition
    pos_type = sofa.SourcePosition_Type
    
    fig = plt.figure(figsize=(15, 5))
    
    if pos_type == 'spherical':
        # Spherical coordinates: azimuth, elevation, radius
        azimuth = positions[:, 0]
        elevation = positions[:, 1]
        radius = positions[:, 2]
        
        if len(positions) == 1:
            # Single source - show clearly
            fig.suptitle('Fixed Source Position', fontsize=14, fontweight='bold')
        
        # Plot 1: Top view (azimuth)
        ax1 = fig.add_subplot(131, projection='polar')
        scatter1 = ax1.scatter(np.deg2rad(azimuth), radius, c=elevation, cmap='viridis', s=200, marker='*')
        ax1.set_title('Top View (Azimuth)')
        plt.colorbar(scatter1, ax=ax1, label='Elevation (deg)')
        
        # Plot 2: 2D scatter (azimuth vs elevation)
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(azimuth, elevation, c=radius, cmap='plasma', s=200, marker='*')
        ax2.set_xlabel('Azimuth (deg)')
        ax2.set_ylabel('Elevation (deg)')
        ax2.set_title('Azimuth vs Elevation')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Radius (m)')
        
        # Plot 3: 3D scatter
        ax3 = fig.add_subplot(133, projection='3d')
        # Convert to Cartesian for 3D plot
        x = radius * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth))
        y = radius * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth))
        z = radius * np.sin(np.deg2rad(elevation))
        ax3.scatter(x, y, z, c='red', s=200, marker='*')
        
        # Add listener at origin
        ax3.scatter([0], [0], [0], c='blue', s=200, marker='o', label='Listener')
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('3D View')
        ax3.legend()
        
        print(f"    Source azimuth: {azimuth[0]:.1f} deg")
        print(f"    Source elevation: {elevation[0]:.1f} deg")
        print(f"    Source distance: {radius[0]:.2f} m")
        
    elif pos_type == 'cartesian':
        # Cartesian coordinates: x, y, z
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        # 3D scatter plot
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='red', s=200, marker='*', label='Source')
        ax.scatter([0], [0], [0], c='blue', s=200, marker='o', label='Listener')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Source Position')
        ax.legend()
        
        print(f"    Source position: ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
    
    plt.tight_layout()
    print("    CHECK Source position plot created")

def get_measurement_indices(sofa, selection):
    """Get list of measurement indices to plot."""
    total_measurements = sofa.Data_IR.shape[0]
    
    if selection == 'all':
        return list(range(total_measurements))
    elif selection == 'first_10':
        return list(range(min(10, total_measurements)))
    elif isinstance(selection, list):
        return [i for i in selection if i < total_measurements]
    else:
        return [0]  # Default to first measurement

def get_position_label(sofa, meas_idx):
    """Get position label for a measurement (handles both source and listener views)."""
    n_sources = len(sofa.SourcePosition)
    has_listener_view = hasattr(sofa, 'ListenerView') and len(sofa.ListenerView) > 1
    
    # Get source position
    source_idx = min(meas_idx, n_sources - 1)
    source_pos = sofa.SourcePosition[source_idx]
    
    # Build label
    if has_listener_view and meas_idx < len(sofa.ListenerView):
        # Show listener view (head orientation)
        listener_view = sofa.ListenerView[meas_idx]
        if sofa.ListenerView_Type == 'spherical':
            label = f'Meas {meas_idx}: Listener Az={listener_view[0]:.0f} deg'
            if n_sources == 1:
                label += f' | Source Az={source_pos[0]:.0f} deg (fixed)'
        else:
            label = f'Meas {meas_idx}: Listener view {meas_idx}'
    else:
        # Show source position
        if sofa.SourcePosition_Type == 'spherical':
            if n_sources == 1:
                label = f'Meas {meas_idx}: Source Az={source_pos[0]:.0f} deg El={source_pos[1]:.0f} deg (fixed)'
            else:
                label = f'Meas {meas_idx}: Source Az={source_pos[0]:.0f} deg El={source_pos[1]:.0f} deg'
        else:
            if n_sources == 1:
                label = f'Meas {meas_idx}: ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}) (fixed)'
            else:
                label = f'Meas {meas_idx}: ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f})'
    
    return label

def plot_impulse_responses(sofa, measurement_indices, plot_left=True, plot_right=True):
    """Plot impulse responses for selected measurements."""
    print("\n" + "="*70)
    print("PLOTTING IMPULSE RESPONSES")
    print("="*70)
    
    fs = sofa.Data_SamplingRate
    n_samples = sofa.Data_IR.shape[2]
    time = np.arange(n_samples) / fs * 1000  # Convert to ms
    
    n_plots = len(measurement_indices)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    if plot_left and plot_right:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    n_receivers = sofa.Data_IR.shape[1]
    
    for idx, meas_idx in enumerate(measurement_indices):
        ax = axes[idx]
        
        # Get IR data
        ir_data = sofa.Data_IR[meas_idx]  # Shape: (receivers, samples)
        
        if plot_left:
            ax.plot(time, ir_data[0, :], label='Left ear', alpha=0.7, linewidth=0.5)
        if plot_right and n_receivers > 1:
            ax.plot(time, ir_data[1, :], label='Right ear', alpha=0.7, linewidth=0.5)
        
        # Get position label
        title = get_position_label(sofa, meas_idx)
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    print(f"  CHECK Plotted {n_plots} impulse responses")

def plot_magnitude_responses(sofa, measurement_indices, plot_left=True, plot_right=True, freq_range=[20, 20000]):
    """Plot magnitude frequency responses for selected measurements."""
    print("\n" + "="*70)
    print("PLOTTING MAGNITUDE RESPONSES")
    print("="*70)
    
    fs = sofa.Data_SamplingRate
    n_receivers = sofa.Data_IR.shape[1]
    
    n_plots = len(measurement_indices)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, meas_idx in enumerate(measurement_indices):
        ax = axes[idx]
        
        # Get IR data and convert to pyfar Signal
        ir_data = sofa.Data_IR[meas_idx]
        
        if plot_left:
            signal_left = pf.Signal(ir_data[0, :], fs)
            freqs = signal_left.frequencies
            mag_left = 20 * np.log10(np.abs(signal_left.freq).flatten() + 1e-10)
            ax.semilogx(freqs, mag_left, label='Left ear', alpha=0.7, linewidth=1)
        
        if plot_right and n_receivers > 1:
            signal_right = pf.Signal(ir_data[1, :], fs)
            mag_right = 20 * np.log10(np.abs(signal_right.freq).flatten() + 1e-10)
            ax.semilogx(freqs, mag_right, label='Right ear', alpha=0.7, linewidth=1)
        
        # Get position label
        title = get_position_label(sofa, meas_idx)
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(freq_range)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    print(f"  CHECK Plotted {n_plots} magnitude responses")

def plot_spectrograms(sofa, measurement_indices, plot_left=True, plot_right=True):
    """Plot spectrograms for selected measurements."""
    print("\n" + "="*70)
    print("PLOTTING SPECTROGRAMS")
    print("="*70)
    
    fs = sofa.Data_SamplingRate
    n_receivers = sofa.Data_IR.shape[1]
    
    for meas_idx in measurement_indices[:5]:  # Limit to 5 for spectrograms
        ir_data = sofa.Data_IR[meas_idx]
        
        n_plots = (1 if plot_left else 0) + (1 if plot_right and n_receivers > 1 else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        if plot_left:
            ax = axes[plot_idx]
            signal_left = pf.Signal(ir_data[0, :], fs)
            
            # Calculate spectrogram
            f, t, Sxx = signal_left.spectrogram(window='hann', window_length=512, overlap=0.5)
            
            im = ax.pcolormesh(t*1000, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (ms)')
            ax.set_title('Left Ear Spectrogram')
            ax.set_ylim([0, fs/2])
            plt.colorbar(im, ax=ax, label='Magnitude (dB)')
            plot_idx += 1
        
        if plot_right and n_receivers > 1:
            ax = axes[plot_idx]
            signal_right = pf.Signal(ir_data[1, :], fs)
            
            f, t, Sxx = signal_right.spectrogram(window='hann', window_length=512, overlap=0.5)
            
            im = ax.pcolormesh(t*1000, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (ms)')
            ax.set_title('Right Ear Spectrogram')
            ax.set_ylim([0, fs/2])
            plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        
        # Get position label for title
        title = get_position_label(sofa, meas_idx)
        fig.suptitle(title)
        
        plt.tight_layout()
    
    print(f"  CHECK Plotted spectrograms for {min(len(measurement_indices), 5)} measurements")

def main():
    """Main function."""
    # Load SOFA file
    sofa = load_sofa_file(SOFA_FILE, skip_verification=SKIP_VERIFICATION)
    if sofa is None:
        return
    
    # Display metadata
    if SHOW_METADATA:
        display_metadata(sofa)
    
    # Get measurement indices to plot
    meas_indices = get_measurement_indices(sofa, MEASUREMENTS_TO_PLOT)
    print(f"\nWill plot {len(meas_indices)} measurements: {meas_indices}")
    
    # Create visualizations
    if SHOW_POSITIONS:
        plot_source_positions(sofa)
    
    if SHOW_IMPULSE_RESPONSES:
        plot_impulse_responses(sofa, meas_indices, PLOT_LEFT_EAR, PLOT_RIGHT_EAR)
    
    if SHOW_MAGNITUDE:
        plot_magnitude_responses(sofa, meas_indices, PLOT_LEFT_EAR, PLOT_RIGHT_EAR, FREQ_RANGE)
    
    if SHOW_SPECTROGRAM:
        plot_spectrograms(sofa, meas_indices, PLOT_LEFT_EAR, PLOT_RIGHT_EAR)
    
    plt.show()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()