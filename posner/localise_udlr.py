import os
from pathlib import Path

import numpy as np
import pyfar as pf
import sofar as sofa
import soundfile as sf


# Input / output config
BASE_DIR = Path(__file__).resolve().parent
INPUT_AUDIO = BASE_DIR / "audio_stimuli" / "pink_noise_48k_30s_300_8000hz.wav"
C2M_SOFA_PATH = Path(r"C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\360-BRIR-FOAIR-database\Binaural\SOFA\C2m.sofa")
C6M_SOFA_PATH = Path(r"C:\Users\tim_e\source\repos\auditory_distance\resources\sofa_files\360-BRIR-FOAIR-database\Binaural\SOFA\C6m.sofa")
SOFA_PATH = C6M_SOFA_PATH
OUTPUT_DIR = INPUT_AUDIO.parent

INPUT_DURATION_SEC = 0.5
OUTPUT_DURATION_SEC = 1.0

# Output mode:
# - "udlr": generate 4 files for Posner labels
# - "angles": generate one file per azimuth in AZIMUTHS_TO_RENDER
OUTPUT_MODE = "udlr"

# For OUTPUT_MODE="angles", set any azimuth list you want (e.g., range(0, 360, 5))
AZIMUTHS_TO_RENDER = list(range(0, 360, 5))
TARGET_ELEVATION_DEG = 0.0

# For OUTPUT_MODE="udlr", map labels to azimuths (horizontal plane)
UDLR_AZIMUTHS = {
    "left": -90.0,
    "right": 90.0,
    "up": 0,# this should be considered "far"
    "down": 0,#this should be considered "near"
}

UDLR_SOFA_BY_LABEL = {
    "left": C2M_SOFA_PATH,
    "right": C2M_SOFA_PATH,
    "down": C2M_SOFA_PATH,
    "up": C6M_SOFA_PATH,
}


def normalize_azimuth_deg(azimuth_deg: float) -> float:
    """Normalize azimuth to [-180, 180)."""
    return ((azimuth_deg + 180.0) % 360.0) - 180.0


def load_mono_signal(audio_path: Path) -> pf.Signal:
    audio, fs_audio = sf.read(str(audio_path))
    audio = np.asarray(audio)

    if audio.ndim == 1:
        mono = audio[np.newaxis, :]
    else:
        mono = np.mean(audio, axis=1, keepdims=False)[np.newaxis, :]

    return pf.Signal(mono, sampling_rate=fs_audio)


def trim_input_duration(audio_pf: pf.Signal, duration_sec: float) -> pf.Signal:
    target_samples = max(1, int(round(duration_sec * audio_pf.sampling_rate)))
    audio_time = np.asarray(audio_pf.time)
    trimmed = audio_time[..., :target_samples]
    if trimmed.shape[-1] < target_samples:
        print(
            f"WARNING: Input shorter than requested {duration_sec:.3f}s; "
            f"using {trimmed.shape[-1] / audio_pf.sampling_rate:.3f}s"
        )
    return pf.Signal(trimmed, sampling_rate=audio_pf.sampling_rate)


def get_sources(sofa_data):
    source_positions = sofa_data.SourcePosition

    azimuth_rad = np.deg2rad(source_positions[:, 0])
    elevation_rad = np.deg2rad(source_positions[:, 1])
    radius = source_positions[:, 2]

    sources = pf.Coordinates.from_spherical_elevation(azimuth_rad, elevation_rad, radius)
    return sources, source_positions


def get_default_distance_m(source_positions: np.ndarray) -> float:
    unique_r = np.unique(np.round(source_positions[:, 2].astype(float), 6))
    if len(unique_r) == 1:
        return float(unique_r[0])
    # If multiple radii exist, default to median and let nearest-neighbor pick exact point
    return float(np.median(unique_r))


def load_sofa_bundle(sofa_path: Path):
    print(f"Loading SOFA: {sofa_path}")
    sofa_data = sofa.read_sofa(str(sofa_path), verify=False)
    sources, source_positions = get_sources(sofa_data)
    default_distance_m = get_default_distance_m(source_positions)
    return {
        "sofa_data": sofa_data,
        "sources": sources,
        "source_positions": source_positions,
        "default_distance_m": default_distance_m,
    }


def get_nearest_brir(sofa_data, sources, source_positions, azimuth_deg: float, elevation_deg: float, distance_m: float):
    target = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(azimuth_deg),
        np.deg2rad(elevation_deg),
        distance_m,
    )

    index, distance = sources.find_nearest(target)
    idx = int(index[0])

    ir_data = sofa_data.Data_IR[idx]
    sampling_rate_ir = (
        sofa_data.Data_SamplingRate[0]
        if hasattr(sofa_data.Data_SamplingRate, "__len__")
        else sofa_data.Data_SamplingRate
    )

    matched_azi = float(source_positions[idx, 0])
    matched_ele = float(source_positions[idx, 1])
    matched_r = float(source_positions[idx, 2])

    print(
        f"Using BRIR idx {idx} for target ({azimuth_deg:.1f}, {elevation_deg:.1f}, {distance_m:.2f}m) -> "
        f"matched ({matched_azi:.1f}, {matched_ele:.1f}, {matched_r:.2f}m), nearest distance: {float(distance):.4f} m"
    )

    ir_pf = pf.Signal(ir_data, sampling_rate_ir, fft_norm="none")
    return ir_pf, idx, matched_azi, matched_ele, matched_r


def localise_to_target(audio_pf: pf.Signal, ir_pf: pf.Signal) -> pf.Signal:
    if audio_pf.sampling_rate != ir_pf.sampling_rate:
        audio_pf = pf.dsp.resample(audio_pf, ir_pf.sampling_rate)

    return pf.dsp.convolve(audio_pf, ir_pf)


def to_stereo_frames(signal: pf.Signal) -> np.ndarray:
    t = np.asarray(signal.time)
    t = np.squeeze(t)

    if t.ndim == 2 and t.shape[0] == 2:
        return t.T
    if t.ndim == 2 and t.shape[1] == 2:
        return t
    if t.ndim == 3:
        if t.shape[0] == 2:
            return t[:, :, 0].T
        if t.shape[1] == 2:
            return t[:, :2, 0]
        if t.shape[2] == 2:
            return t[:, 0, :]

    raise ValueError(f"Could not extract stereo channels from convolved signal with shape {t.shape}")


def save_binaural(output_path: Path, binaural: pf.Signal, output_duration_sec: float = None):
    stereo_frames = to_stereo_frames(binaural)
    if stereo_frames.ndim != 2 or stereo_frames.shape[1] != 2:
        raise ValueError(f"Output is not stereo. Shape: {stereo_frames.shape}")

    if output_duration_sec is not None:
        target_samples = max(1, int(round(output_duration_sec * binaural.sampling_rate)))
        if stereo_frames.shape[0] < target_samples:
            pad = target_samples - stereo_frames.shape[0]
            stereo_frames = np.pad(stereo_frames, ((0, pad), (0, 0)), mode="constant")
        else:
            stereo_frames = stereo_frames[:target_samples, :]

    sf.write(str(output_path), stereo_frames, int(binaural.sampling_rate))
    print(f"Saved: {output_path} (shape={stereo_frames.shape})")


def render_udlr(audio_pf, input_stem, sofa_bundle_by_label):
    used_indices = set()

    for label, az in UDLR_AZIMUTHS.items():
        az = normalize_azimuth_deg(az)
        bundle = sofa_bundle_by_label[label]
        distance_m = bundle["default_distance_m"]

        print(f"\nProcessing target: {label} @ azimuth {az:.1f}°, distance {distance_m:.2f}m")
        ir_pf, idx, *_ = get_nearest_brir(
            bundle["sofa_data"],
            bundle["sources"],
            bundle["source_positions"],
            az,
            TARGET_ELEVATION_DEG,
            distance_m,
        )
        if idx in used_indices:
            print(f"WARNING: BRIR idx {idx} already used by another label")
        used_indices.add(idx)

        binaural = localise_to_target(audio_pf, ir_pf)
        output_path = OUTPUT_DIR / f"{input_stem}_{label}.wav"
        save_binaural(output_path, binaural, output_duration_sec=OUTPUT_DURATION_SEC)


def render_angles(audio_pf, sofa_data, sources, source_positions, input_stem, distance_m):
    used_indices = {}
    for az in AZIMUTHS_TO_RENDER:
        az_n = normalize_azimuth_deg(float(az))
        print(f"\nProcessing azimuth: {az_n:.1f}°")
        ir_pf, idx, matched_azi, _, _ = get_nearest_brir(
            sofa_data, sources, source_positions, az_n, TARGET_ELEVATION_DEG, distance_m
        )
        if idx in used_indices:
            print(
                f"WARNING: azimuth {az_n:.1f}° maps to same BRIR idx {idx} as {used_indices[idx]:.1f}°"
            )
        used_indices[idx] = az_n

        binaural = localise_to_target(audio_pf, ir_pf)
        output_path = OUTPUT_DIR / f"{input_stem}_az{int(round(az_n)):03d}_matched{int(round(matched_azi)):03d}.wav"
        save_binaural(output_path, binaural, output_duration_sec=OUTPUT_DURATION_SEC)


def main():
    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(f"Input audio not found: {INPUT_AUDIO}")

    print(f"Loading input audio: {INPUT_AUDIO}")
    audio_pf = load_mono_signal(INPUT_AUDIO)
    audio_pf = trim_input_duration(audio_pf, INPUT_DURATION_SEC)
    print(f"Using input duration: {audio_pf.n_samples / audio_pf.sampling_rate:.3f}s")

    input_stem = INPUT_AUDIO.stem

    if OUTPUT_MODE == "udlr":
        sofa_paths = set(UDLR_SOFA_BY_LABEL.values())
        for p in sofa_paths:
            if not p.exists():
                raise FileNotFoundError(f"SOFA file not found: {p}")

        sofa_bundle_by_path = {p: load_sofa_bundle(p) for p in sofa_paths}
        sofa_bundle_by_label = {label: sofa_bundle_by_path[path] for label, path in UDLR_SOFA_BY_LABEL.items()}

        for label, path in UDLR_SOFA_BY_LABEL.items():
            print(f"UDLR mapping: {label} -> {path.name}")

        render_udlr(audio_pf, input_stem, sofa_bundle_by_label)
    elif OUTPUT_MODE == "angles":
        if not SOFA_PATH.exists():
            raise FileNotFoundError(f"SOFA file not found: {SOFA_PATH}")
        bundle = load_sofa_bundle(SOFA_PATH)
        print(f"Using default distance: {bundle['default_distance_m']:.2f} m")
        render_angles(
            audio_pf,
            bundle["sofa_data"],
            bundle["sources"],
            bundle["source_positions"],
            input_stem,
            bundle["default_distance_m"],
        )
    else:
        raise ValueError(f"Unsupported OUTPUT_MODE: {OUTPUT_MODE}")

    print("\nDone.")


if __name__ == "__main__":
    main()
