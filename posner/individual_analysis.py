#calculte mean response times for each condition 
#file:"C:\Users\tim_e\source\repos\auditory_distance\posner\results\posner_pilot_20260326_170237.csv"
#headers are trial_number	sound_location	dot_location	is_valid	sound_file	response	correct	response_time
#calculate mean response times for each condition (valid/invalid) and sound location (up down left right) so eight conditions. also calculate mean valid response time and mean invalid response time as well as mean response times for each location

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


CSV_PATH = Path(r"C:\Users\tim_e\source\repos\auditory_distance\posner\results\posner_pilot_20260327_102641.csv")
LOCATIONS = ("up", "down", "left", "right")
VALIDITY = (True, False)


def parse_bool(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "valid"}:
        return True
    if v in {"false", "0", "no", "n", "invalid"}:
        return False
    return None


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def load_trials(csv_path: Path) -> List[Tuple[str, bool, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[Tuple[str, bool, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            location = str(row.get("sound_location", "")).strip().lower()
            is_valid = parse_bool(str(row.get("is_valid", "")))
            rt = safe_float(str(row.get("response_time", "")))

            if location not in LOCATIONS:
                continue
            if is_valid is None or rt is None:
                continue

            rows.append((location, is_valid, rt))

    return rows


def run_analysis(csv_path: Path) -> None:
    rows = load_trials(csv_path)
    if not rows:
        print("No valid rows found for analysis.")
        return

    by_condition: Dict[Tuple[bool, str], List[float]] = {
        (valid, loc): [] for valid in VALIDITY for loc in LOCATIONS
    }
    by_validity: Dict[bool, List[float]] = {True: [], False: []}
    by_location: Dict[str, List[float]] = {loc: [] for loc in LOCATIONS}

    for location, is_valid, rt in rows:
        by_condition[(is_valid, location)].append(rt)
        by_validity[is_valid].append(rt)
        by_location[location].append(rt)

    print(f"Analyzed file: {csv_path}")
    print(f"Valid trials used: {len(rows)}")

    print("\nMean response time by condition (validity x sound_location):")
    for valid in VALIDITY:
        valid_label = "valid" if valid else "invalid"
        for loc in LOCATIONS:
            m = mean(by_condition[(valid, loc)])
            n = len(by_condition[(valid, loc)])
            m_str = f"{m:.3f}" if m is not None else "NA"
            print(f"  {valid_label:7s} | {loc:5s} -> mean_rt={m_str}s (n={n})")

    print("\nMean response time by validity:")
    for valid in VALIDITY:
        valid_label = "valid" if valid else "invalid"
        m = mean(by_validity[valid])
        n = len(by_validity[valid])
        m_str = f"{m:.3f}" if m is not None else "NA"
        print(f"  {valid_label:7s} -> mean_rt={m_str}s (n={n})")

    print("\nMean response time by sound location:")
    for loc in LOCATIONS:
        m = mean(by_location[loc])
        n = len(by_location[loc])
        m_str = f"{m:.3f}" if m is not None else "NA"
        print(f"  {loc:5s} -> mean_rt={m_str}s (n={n})")


if __name__ == "__main__":
    run_analysis(CSV_PATH)
