# -*- coding: utf-8 -*-
# Pilot Data Analysis Script
# Analyzes accuracy results from pilot experiment CSV files
# Includes proper 2-way ANOVA for presentation method x stimulus type

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
script_dir = Path(__file__).resolve().parent
results_folder = Path(r"C:\Users\tim_e\source\repos\auditory_distance\results\pilot_26_01")

# Create analysis results folder (MOVE THIS HERE)
analysis_results_folder = script_dir / 'analysis_results'
analysis_results_folder.mkdir(parents=True, exist_ok=True)

# Check if results folder exists
if not results_folder.exists():
    print(f"Error: Results folder not found: {results_folder}")
    exit(1)

# Create empty DataFrame for analysis results
analysis_results = pd.DataFrame(columns=['participant_id', 'correct_percent', 'correct_count', 'total_trials'])

print("Analyzing pilot experiment results...")
print("=" * 70)

# Get only trial CSV files (not demographics)
csv_files = [f for f in results_folder.glob('*_trials.csv')]

if not csv_files:
    print(f"No trial CSV files found in {results_folder}")
    exit(1)

print(f"Found {len(csv_files)} participant file(s)\n")

# Process each CSV file
for csv_file in csv_files:
    participant_id = csv_file.stem.replace('_trials', '')  # Remove _trials suffix
    
    print(f"Processing: {participant_id}")
    
    try:
        # Load CSV data
        data = pd.read_csv(csv_file)
        
        # Filter to experimental trials only
        experimental_data = data[data['trial_type'] == 'experimental']
        
        # Count correct answers (overall)
        correct_answers = experimental_data['accuracy'].sum()
        total_trials = len(experimental_data)
        
        # Calculate percentage (overall)
        if total_trials > 0:
            correct_answers_percent = (correct_answers / total_trials) * 100
        else:
            correct_answers_percent = 0
            print(f"  WARNING: No experimental trials found for {participant_id}")
        
        # Calculate mean accuracy for each of the six conditions
        condition_accuracies = experimental_data.groupby(
            ['presentation_type', 'stimulus_category']
        )['accuracy'].mean()
        
        # Calculate mean accuracy by presentation type (collapsed across stimuli)
        presentation_accuracies = experimental_data.groupby('presentation_type')['accuracy'].mean()
        
        # Calculate mean accuracy by stimulus type (collapsed across presentation)
        stimulus_accuracies = experimental_data.groupby('stimulus_category')['accuracy'].mean()
        
        # Create result row with overall stats
        result_row = {
            'participant_id': participant_id,
            'correct_percent': correct_answers_percent,
            'correct_count': correct_answers,
            'total_trials': total_trials,
            # Add collapsed by presentation type
            'all_headphone_percent': presentation_accuracies.get('headphone', np.nan) * 100 if 'headphone' in presentation_accuracies.index else np.nan,
            'all_speaker_percent': presentation_accuracies.get('speaker', np.nan) * 100 if 'speaker' in presentation_accuracies.index else np.nan,
            # Add collapsed by stimulus type
            'all_environment_percent': stimulus_accuracies.get('environment', np.nan) * 100 if 'environment' in stimulus_accuracies.index else np.nan,
            'all_ISTS_percent': stimulus_accuracies.get('ISTS', np.nan) * 100 if 'ISTS' in stimulus_accuracies.index else np.nan,
            'all_noise_percent': stimulus_accuracies.get('noise', np.nan) * 100 if 'noise' in stimulus_accuracies.index else np.nan,
        }
        
        # Add condition-specific accuracies (convert to percentage)
        conditions = [
            ('headphone', 'environment'),
            ('headphone', 'ISTS'),
            ('headphone', 'noise'),
            ('speaker', 'environment'),
            ('speaker', 'ISTS'),
            ('speaker', 'noise')
        ]
        
        for presentation, stimulus in conditions:
            col_name = f'{presentation}_{stimulus}_percent'
            try:
                accuracy = condition_accuracies.loc[(presentation, stimulus)] * 100
                result_row[col_name] = accuracy
            except KeyError:
                # If condition not found (shouldn't happen with balanced design)
                result_row[col_name] = np.nan
        
        # Add to analysis results
        analysis_results = pd.concat([analysis_results, pd.DataFrame([result_row])], ignore_index=True)
        
        print(f"  Overall: {correct_answers}/{total_trials} ({correct_answers_percent:.1f}%)")
        
    except Exception as e:
        print(f"  Error processing {participant_id}: {e}")
        continue

print("\n" + "=" * 70)
print("Analysis Summary:")
print(analysis_results.to_string(index=False))

# Calculate overall statistics
if len(analysis_results) > 0:
    mean_accuracy = analysis_results['correct_percent'].mean()
    std_accuracy = analysis_results['correct_percent'].std()
    print(f"\nMean accuracy: {mean_accuracy:.1f}% (SD: {std_accuracy:.1f}%)")

# Save analysis results as CSV
output_file = analysis_results_folder / 'pilot_analysis_results.csv'
analysis_results.to_csv(output_file, index=False)
print(f"\nAnalysis results saved to: {output_file}")

# ============================================================================
# DETAILED STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("DETAILED STATISTICAL ANALYSIS")
print("=" * 70)

# Combine all experimental data from all participants
all_data = []
for csv_file in csv_files:
    try:
        data = pd.read_csv(csv_file)
        # Filter to experimental trials only (FIXED)
        experimental_data = data[data['trial_type'] == 'experimental']
        experimental_data['participant_id'] = csv_file.stem.replace('_trials', '')
        all_data.append(experimental_data)
    except Exception as e:
        print(f"Error loading {csv_file.name}: {e}")
        continue

if not all_data:
    print("No data available for analysis")
    exit(1)

# Combine into single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

print(f"\nTotal trials analyzed: {len(combined_data)}")
print(f"Participants: {combined_data['participant_id'].nunique()}")

# ============================================================================
# DESCRIPTIVE STATISTICS BY CONDITION (FIXED - Participant-level SDs)
# ============================================================================

print("\n" + "-" * 70)
print("DESCRIPTIVE STATISTICS")
print("-" * 70)

# First, calculate each participant's mean accuracy for each condition
participant_means = combined_data.groupby(
    ['participant_id', 'presentation_type', 'stimulus_category']
)['accuracy'].mean().reset_index()

# Now calculate statistics ACROSS PARTICIPANTS for each condition
condition_stats = participant_means.groupby(['presentation_type', 'stimulus_category'])['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('sem', 'sem'),
    ('count', 'count')
]).reset_index()

condition_stats['mean_percent'] = condition_stats['mean'] * 100
condition_stats['std_percent'] = condition_stats['std'] * 100

print("\nAccuracy by Presentation Method and Stimulus Type:")
print("(Mean ± SD across participants)")
print(condition_stats.to_string(index=False))

# Overall means by presentation type
print("\n" + "-" * 70)
print("Mean Accuracy by Presentation Method:")
participant_presentation = participant_means.groupby(
    ['participant_id', 'presentation_type']
)['accuracy'].mean().reset_index()

presentation_means = participant_presentation.groupby('presentation_type')['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('sem', 'sem'),
    ('count', 'count')
])
presentation_means['mean_percent'] = presentation_means['mean'] * 100
presentation_means['std_percent'] = presentation_means['std'] * 100
print(presentation_means)

# Overall means by stimulus type
print("\n" + "-" * 70)
print("Mean Accuracy by Stimulus Type:")
participant_stimulus = participant_means.groupby(
    ['participant_id', 'stimulus_category']
)['accuracy'].mean().reset_index()

stimulus_means = participant_stimulus.groupby('stimulus_category')['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('sem', 'sem'),
    ('count', 'count')
])
stimulus_means['mean_percent'] = stimulus_means['mean'] * 100
stimulus_means['std_percent'] = stimulus_means['std'] * 100
print(stimulus_means)

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================

# Save condition statistics
condition_stats_file = analysis_results_folder / 'condition_statistics.csv'
condition_stats.to_csv(condition_stats_file, index=False)
print(f"\n" + "=" * 70)
print(f"Condition statistics saved to: {condition_stats_file}")

# Save combined trial data
combined_data_file = analysis_results_folder / 'combined_trial_data.csv'
combined_data.to_csv(combined_data_file, index=False)
print(f"Combined trial data saved to: {combined_data_file}")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)

# ============================================================================
# REPEATED MEASURES 2-WAY ANOVA (pingouin)
# ============================================================================

print("\n" + "=" * 70)
print("REPEATED MEASURES 2-WAY ANOVA: Presentation Method x Stimulus Type")
print("=" * 70)

try:
    import pingouin as pg
    
    print("\nRunning repeated measures ANOVA with pingouin...")
    
    # Run two-way repeated measures ANOVA
    aov = pg.rm_anova(
        data=participant_means,
        dv='accuracy',
        within=['presentation_type', 'stimulus_category'],
        subject='participant_id',
        detailed=True
    )
    
    print("\nRepeated Measures ANOVA Table:")
    print(aov.to_string(index=False))
    
    # Extract results
    p_presentation = aov.loc[aov['Source'] == 'presentation_type', 'p-unc'].values[0]
    p_stimulus = aov.loc[aov['Source'] == 'stimulus_category', 'p-unc'].values[0]
    p_interaction = aov.loc[aov['Source'] == 'presentation_type * stimulus_category', 'p-unc'].values[0]
    
    f_presentation = aov.loc[aov['Source'] == 'presentation_type', 'F'].values[0]
    f_stimulus = aov.loc[aov['Source'] == 'stimulus_category', 'F'].values[0]
    f_interaction = aov.loc[aov['Source'] == 'presentation_type * stimulus_category', 'F'].values[0]
    
    df1_presentation = aov.loc[aov['Source'] == 'presentation_type', 'ddof1'].values[0]
    df2_presentation = aov.loc[aov['Source'] == 'presentation_type', 'ddof2'].values[0]
    df1_stimulus = aov.loc[aov['Source'] == 'stimulus_category', 'ddof1'].values[0]
    df2_stimulus = aov.loc[aov['Source'] == 'stimulus_category', 'ddof2'].values[0]
    df1_interaction = aov.loc[aov['Source'] == 'presentation_type * stimulus_category', 'ddof1'].values[0]
    df2_interaction = aov.loc[aov['Source'] == 'presentation_type * stimulus_category', 'ddof2'].values[0]
    
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    
    print(f"\nMain Effect: Presentation Method")
    print(f"  F({df1_presentation:.0f}, {df2_presentation:.0f}) = {f_presentation:.3f}")
    print(f"  p-value: {p_presentation:.4f}")
    print(f"  Result: {'SIGNIFICANT' if p_presentation < 0.05 else 'Not significant'} (α = 0.05)")
    
    print(f"\nMain Effect: Stimulus Type")
    print(f"  F({df1_stimulus:.0f}, {df2_stimulus:.0f}) = {f_stimulus:.3f}")
    print(f"  p-value: {p_stimulus:.4f}")
    print(f"  Result: {'SIGNIFICANT' if p_stimulus < 0.05 else 'Not significant'} (α = 0.05)")
    
    print(f"\nInteraction Effect: Presentation × Stimulus")
    print(f"  F({df1_interaction:.0f}, {df2_interaction:.0f}) = {f_interaction:.3f}")
    print(f"  p-value: {p_interaction:.4f}")
    print(f"  Result: {'SIGNIFICANT' if p_interaction < 0.05 else 'Not significant'} (α = 0.05)")
    
    # Check for sphericity violations
    print("\n" + "-" * 70)
    print("SPHERICITY CHECK:")
    print("-" * 70)
    for idx, row in aov.iterrows():
        if pd.notna(row.get('p-spher')):
            source = row['Source']
            p_spher = row['p-spher']
            print(f"\n{source}:")
            print(f"  Sphericity p-value: {p_spher:.4f}")
            if p_spher < 0.05:
                print(f"  WARNING: Sphericity assumption violated (p < 0.05)")
                print(f"  Consider using Greenhouse-Geisser correction: p-GG-corr = {row.get('p-GG-corr', 'N/A')}")
            else:
                print(f"  Sphericity assumption met (p >= 0.05)")
    
    # Save ANOVA table
    anova_output_file = analysis_results_folder / 'repeated_measures_anova.csv'
    aov.to_csv(anova_output_file, index=False)
    print(f"\n  ANOVA table saved to: {anova_output_file}")
    
except ImportError:
    print("Error: pingouin not installed. Install with: pip install pingouin")
except Exception as e:
    print(f"Error performing repeated measures ANOVA: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# INSPECT DATA FOR REPEATED MEASURES ANOVA
# ============================================================================

print("\n" + "=" * 70)
print("DATA USED FOR REPEATED MEASURES ANOVA")
print("=" * 70)

print("\nParticipant means by condition (used in ANOVA):")
print(participant_means.to_string(index=False))

# Create a pivot table for easier viewing
print("\n" + "-" * 70)
print("PIVOT TABLE VIEW (rows=participants, columns=conditions):")
print("-" * 70)

pivot_table = participant_means.pivot_table(
    index='participant_id',
    columns=['presentation_type', 'stimulus_category'],
    values='accuracy'
)
print(pivot_table)

# Save this data
participant_means_file = analysis_results_folder / 'participant_means_for_anova.csv'
participant_means.to_csv(participant_means_file, index=False)
print(f"\nParticipant means saved to: {participant_means_file}")

pivot_file = analysis_results_folder / 'participant_means_pivot.csv'
pivot_table.to_csv(pivot_file)
print(f"Pivot table saved to: {pivot_file}")

# Show data structure
print("\n" + "-" * 70)
print("DATA STRUCTURE:")
print(f"  Total rows: {len(participant_means)}")
print(f"  Participants: {participant_means['participant_id'].nunique()}")
print(f"  Conditions per participant: {len(participant_means) // participant_means['participant_id'].nunique()}")
print(f"\nColumns: {list(participant_means.columns)}")

print("\n" + "-" * 70)
print("CEILING EFFECTS CHECK:")
print("-" * 70)

ceiling_threshold = 95  # Define ceiling as ≥95% correct

ceiling_participants = analysis_results[analysis_results['correct_percent'] >= ceiling_threshold]
n_ceiling = len(ceiling_participants)
total_participants = len(analysis_results)

print(f"\nParticipants at ceiling (≥{ceiling_threshold}%): {n_ceiling}/{total_participants} ({n_ceiling/total_participants*100:.1f}%)")

if n_ceiling > 0:
    print(f"\nCeiling participants:")
    print(ceiling_participants[['participant_id', 'correct_percent']].to_string(index=False))
    print("\nNote: Ceiling effects may reduce power to detect condition differences.")






























