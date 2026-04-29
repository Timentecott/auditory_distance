# Pilot Data Demographics Analysis Script
# Analyzes demographic data from pilot experiment CSV files

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Specify the folder containing pilot data
script_dir = Path(__file__).resolve().parent
results_folder = Path(r"C:\Users\tim_e\source\repos\auditory_distance\results\pilot_26_01")

# Create analysis results folder
analysis_results_folder = script_dir / 'analysis_results'
analysis_results_folder.mkdir(parents=True, exist_ok=True)

# Check if results folder exists
if not results_folder.exists():
    print(f"Error: Results folder not found: {results_folder}")
    exit(1)

print("=" * 70)
print("PILOT DEMOGRAPHICS ANALYSIS")
print("=" * 70)
print(f"\nResults folder: {results_folder}")

# ============================================================================
# LOAD DEMOGRAPHICS DATA
# ============================================================================

# Identify CSV files with extension _demographics.csv
demographics_files = list(results_folder.glob('*_demographics.csv'))

if not demographics_files:
    print(f"\nError: No demographics CSV files found in {results_folder}")
    exit(1)

print(f"\nFound {len(demographics_files)} participant demographics file(s)\n")

# Load each CSV and extract demographics data
all_demographics = []

for csv_file in demographics_files:
    try:
        print(f"Loading: {csv_file.name}")
        
        # Load CSV data
        data = pd.read_csv(csv_file)
        
        # Extract required columns
        demographics_data = data[['participant_id', 'age', 'gender', 'ethnicity', 
                                  'hearing_problems', 'musician', 'musical_experience']]
        
        all_demographics.append(demographics_data)
        
    except Exception as e:
        print(f"  Error loading {csv_file.name}: {e}")
        continue

if not all_demographics:
    print("\nError: No demographics data could be loaded")
    exit(1)

# Compile the extracted data into a single DataFrame
compiled_demographics = pd.concat(all_demographics, ignore_index=True)

print(f"\n{len(compiled_demographics)} participant(s) loaded successfully")

# ============================================================================
# CALCULATE SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("DEMOGRAPHICS SUMMARY")
print("=" * 70)

# Convert age to numeric (handle any non-numeric entries)
compiled_demographics['age_numeric'] = pd.to_numeric(compiled_demographics['age'], errors='coerce')

# Calculate mean age and SD
mean_age = compiled_demographics['age_numeric'].mean()
sd_age = compiled_demographics['age_numeric'].std()

print(f"\nAge Statistics:")
print(f"  Mean: {mean_age:.1f} years")
print(f"  SD: {sd_age:.1f} years")
print(f"  Range: {compiled_demographics['age_numeric'].min():.0f} - {compiled_demographics['age_numeric'].max():.0f} years")

# Count of gender
print(f"\nGender Distribution:")
gender_counts = compiled_demographics['gender'].value_counts()
for gender, count in gender_counts.items():
    percentage = (count / len(compiled_demographics)) * 100
    print(f"  {gender}: {count} ({percentage:.1f}%)")

# Count of ethnicity
print(f"\nEthnicity Distribution:")
ethnicity_counts = compiled_demographics['ethnicity'].value_counts()
for ethnicity, count in ethnicity_counts.items():
    percentage = (count / len(compiled_demographics)) * 100
    print(f"  {ethnicity}: {count} ({percentage:.1f}%)")

# Count of hearing problems
print(f"\nHearing Problems:")
hearing_counts = compiled_demographics['hearing_problems'].value_counts()
for response, count in hearing_counts.items():
    percentage = (count / len(compiled_demographics)) * 100
    # Check if response indicates no hearing problems
    has_problem = response.lower() not in ['no', 'none', 'n/a', 'na']
    status = "Yes" if has_problem else "No"
    print(f"  {status}: {count} ({percentage:.1f}%)")

# Count of musicians
print(f"\nMusician Status:")
musician_counts = compiled_demographics['musician'].value_counts()
for response, count in musician_counts.items():
    percentage = (count / len(compiled_demographics)) * 100
    print(f"  {response}: {count} ({percentage:.1f}%)")

# ============================================================================
# CREATE SUMMARY STATISTICS DATAFRAME
# ============================================================================

# Create summary statistics dictionary
summary_stats = {
    'Statistic': [
        'Total Participants',
        'Mean Age',
        'SD Age',
        'Age Range',
        '',
        'Gender Distribution',
    ],
    'Value': [
        len(compiled_demographics),
        f"{mean_age:.1f}",
        f"{sd_age:.1f}",
        f"{compiled_demographics['age_numeric'].min():.0f} - {compiled_demographics['age_numeric'].max():.0f}",
        '',
        '',
    ]
}

# Add gender counts
for gender, count in gender_counts.items():
    summary_stats['Statistic'].append(f"  {gender}")
    summary_stats['Value'].append(f"{count} ({(count/len(compiled_demographics)*100):.1f}%)")

summary_stats['Statistic'].append('')
summary_stats['Value'].append('')
summary_stats['Statistic'].append('Ethnicity Distribution')
summary_stats['Value'].append('')

# Add ethnicity counts
for ethnicity, count in ethnicity_counts.items():
    summary_stats['Statistic'].append(f"  {ethnicity}")
    summary_stats['Value'].append(f"{count} ({(count/len(compiled_demographics)*100):.1f}%)")

summary_stats['Statistic'].append('')
summary_stats['Value'].append('')
summary_stats['Statistic'].append('Musicians')
summary_stats['Value'].append('')

# Add musician counts
for response, count in musician_counts.items():
    summary_stats['Statistic'].append(f"  {response}")
    summary_stats['Value'].append(f"{count} ({(count/len(compiled_demographics)*100):.1f}%)")

# Create DataFrame
summary_df = pd.DataFrame(summary_stats)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save compiled demographics DataFrame
compiled_file = analysis_results_folder / 'compiled_demographics.csv'
compiled_demographics.to_csv(compiled_file, index=False)
print(f"\nCompiled demographics saved to: {compiled_file}")

# Save summary statistics
summary_file = analysis_results_folder / 'demographics_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"Summary statistics saved to: {summary_file}")

# Display compiled demographics
print("\n" + "=" * 70)
print("COMPILED DEMOGRAPHICS")
print("=" * 70)
print(compiled_demographics.to_string(index=False))

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)
