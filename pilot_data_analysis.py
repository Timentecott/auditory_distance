# Pilot Data Analysis Script
# Analyzes accuracy results from pilot experiment CSV files
# Includes 2-way ANOVA for presentation method x stimulus type

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = Path(__file__).resolve().parent
results_folder = script_dir / 'results'

# Check if results folder exists
if not results_folder.exists():
    print(f"Error: Results folder not found: {results_folder}")
    exit(1)

# Create empty DataFrame for analysis results
analysis_results = pd.DataFrame(columns=['participant_id', 'correct_percent', 'correct_count', 'total_trials'])

print("Analyzing pilot experiment results...")
print("=" * 70)

# Get all CSV files in results folder
csv_files = list(results_folder.glob('*.csv'))

if not csv_files:
    print(f"No CSV files found in {results_folder}")
    exit(1)

print(f"Found {len(csv_files)} participant file(s)\n")

# Process each CSV file
for csv_file in csv_files:
    participant_id = csv_file.stem  # Filename without extension
    
    print(f"Processing: {participant_id}")
    
    try:
        # Load CSV data
        data = pd.read_csv(csv_file)
        
        # Filter out practice trials (rows with '(practice)' in presentation_type)
        experimental_data = data[~data['presentation_type'].str.contains('practice', case=False, na=False)]
        
        # Get accuracy column (column index 5, or column name 'accuracy')
        if 'accuracy' in data.columns:
            accuracy_col = experimental_data['accuracy']
        else:
            # Fallback to column index 5 if 'accuracy' column doesn't exist
            accuracy_col = experimental_data.iloc[:, 5]
        
        # Count correct answers (1s) in experimental trials
        correct_answers = accuracy_col.sum()
        total_trials = len(experimental_data)
        
        # Calculate percentage
        if total_trials > 0:
            correct_answers_percent = (correct_answers / total_trials) * 100
        else:
            correct_answers_percent = 0
            print(f"  WARNING: No experimental trials found for {participant_id}")
        
        # Add to analysis results
        analysis_results.loc[len(analysis_results)] = {
            'participant_id': participant_id,
            'correct_percent': correct_answers_percent,
            'correct_count': correct_answers,
            'total_trials': total_trials
        }
        
        print(f"  Correct: {correct_answers}/{total_trials} ({correct_answers_percent:.1f}%)")
        
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
output_file = script_dir / 'pilot_analysis_results.csv'
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
        # Filter out practice trials
        experimental_data = data[~data['presentation_type'].str.contains('practice', case=False, na=False)]
        experimental_data['participant_id'] = csv_file.stem
        all_data.append(experimental_data)
    except Exception as e:
        print(f"Error loading {csv_file.name}: {e}")
        continue

if not all_data:
    print("No data available for analysis")
    exit(1)

# Combine into single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Clean presentation_type (remove any remaining practice labels)
combined_data['presentation_type'] = combined_data['presentation_type'].str.replace('(practice)', '', regex=False).str.strip()

print(f"\nTotal trials analyzed: {len(combined_data)}")
print(f"Participants: {combined_data['participant_id'].nunique()}")

# ============================================================================
# DESCRIPTIVE STATISTICS BY CONDITION
# ============================================================================

print("\n" + "-" * 70)
print("DESCRIPTIVE STATISTICS")
print("-" * 70)

# Calculate mean accuracy by condition
condition_stats = combined_data.groupby(['presentation_type', 'stimulus_category'])['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).reset_index()

condition_stats['mean_percent'] = condition_stats['mean'] * 100
condition_stats['std_percent'] = condition_stats['std'] * 100

print("\nAccuracy by Presentation Method and Stimulus Type:")
print(condition_stats.to_string(index=False))

# Overall means by presentation type
print("\n" + "-" * 70)
print("Mean Accuracy by Presentation Method:")
presentation_means = combined_data.groupby('presentation_type')['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
])
presentation_means['mean_percent'] = presentation_means['mean'] * 100
presentation_means['std_percent'] = presentation_means['std'] * 100
print(presentation_means)

# Overall means by stimulus type
print("\n" + "-" * 70)
print("Mean Accuracy by Stimulus Type:")
stimulus_means = combined_data.groupby('stimulus_category')['accuracy'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
])
stimulus_means['mean_percent'] = stimulus_means['mean'] * 100
stimulus_means['std_percent'] = stimulus_means['std'] * 100
print(stimulus_means)

# ============================================================================
# 2-WAY ANOVA: Presentation Method x Stimulus Type
# ============================================================================

print("\n" + "=" * 70)
print("2-WAY ANOVA: Presentation Method x Stimulus Type")
print("=" * 70)

# Prepare data for ANOVA
# Create numeric codes for factors
combined_data['presentation_code'] = combined_data['presentation_type'].map({'headphone': 0, 'speaker': 1})
combined_data['stimulus_code'] = combined_data['stimulus_category'].map({'environment': 0, 'ISTS': 1, 'noise': 2})

# Get groups for each condition
groups = {}
for presentation in ['headphone', 'speaker']:
    for stimulus in ['environment', 'ISTS', 'noise']:
        condition = f"{presentation}_{stimulus}"
        mask = (combined_data['presentation_type'] == presentation) & (combined_data['stimulus_category'] == stimulus)
        groups[condition] = combined_data[mask]['accuracy'].values

# Check sample sizes
print("\nSample sizes per condition:")
for condition, data in groups.items():
    print(f"  {condition}: n={len(data)}")

# Perform 2-way ANOVA using scipy
# Main effect: Presentation Method
headphone_acc = combined_data[combined_data['presentation_type'] == 'headphone']['accuracy']
speaker_acc = combined_data[combined_data['presentation_type'] == 'speaker']['accuracy']

f_stat_presentation, p_val_presentation = stats.f_oneway(headphone_acc, speaker_acc)

print("\n" + "-" * 70)
print("MAIN EFFECT: Presentation Method (Headphone vs Speaker)")
print(f"  F-statistic: {f_stat_presentation:.4f}")
print(f"  p-value: {p_val_presentation:.4f}")
if p_val_presentation < 0.05:
    print(f"  Result: SIGNIFICANT (p < 0.05)")
else:
    print(f"  Result: Not significant (p >= 0.05)")

# Main effect: Stimulus Type
env_acc = combined_data[combined_data['stimulus_category'] == 'environment']['accuracy']
ists_acc = combined_data[combined_data['stimulus_category'] == 'ISTS']['accuracy']
noise_acc = combined_data[combined_data['stimulus_category'] == 'noise']['accuracy']

f_stat_stimulus, p_val_stimulus = stats.f_oneway(env_acc, ists_acc, noise_acc)

print("\n" + "-" * 70)
print("MAIN EFFECT: Stimulus Type (Environment vs ISTS vs Noise)")
print(f"  F-statistic: {f_stat_stimulus:.4f}")
print(f"  p-value: {p_val_stimulus:.4f}")
if p_val_stimulus < 0.05:
    print(f"  Result: SIGNIFICANT (p < 0.05)")
else:
    print(f"  Result: Not significant (p >= 0.05)")

# Interaction effect (using all 6 conditions)
all_groups = [groups[k] for k in sorted(groups.keys())]
f_stat_interaction, p_val_interaction = stats.f_oneway(*all_groups)

print("\n" + "-" * 70)
print("INTERACTION EFFECT: Presentation Method x Stimulus Type")
print(f"  F-statistic: {f_stat_interaction:.4f}")
print(f"  p-value: {p_val_interaction:.4f}")
if p_val_interaction < 0.05:
    print(f"  Result: SIGNIFICANT interaction (p < 0.05)")
else:
    print(f"  Result: No significant interaction (p >= 0.05)")

# ============================================================================
# POST-HOC PAIRWISE COMPARISONS
# ============================================================================

if p_val_stimulus < 0.05:
    print("\n" + "=" * 70)
    print("POST-HOC PAIRWISE COMPARISONS (Stimulus Type)")
    print("=" * 70)
    
    # Pairwise t-tests with Bonferroni correction
    comparisons = [
        ('environment', 'ISTS'),
        ('environment', 'noise'),
        ('ISTS', 'noise')
    ]
    
    alpha_corrected = 0.05 / len(comparisons)  # Bonferroni correction
    print(f"\nBonferroni-corrected alpha: {alpha_corrected:.4f}")
    
    for stim1, stim2 in comparisons:
        data1 = combined_data[combined_data['stimulus_category'] == stim1]['accuracy']
        data2 = combined_data[combined_data['stimulus_category'] == stim2]['accuracy']
        t_stat, p_val = stats.ttest_ind(data1, data2)
        
        mean1 = data1.mean() * 100
        mean2 = data2.mean() * 100
        
        print(f"\n{stim1} vs {stim2}:")
        print(f"  Mean difference: {mean1:.1f}% vs {mean2:.1f}%")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        if p_val < alpha_corrected:
            print(f"  Result: SIGNIFICANT (p < {alpha_corrected:.4f})")
        else:
            print(f"  Result: Not significant")

# ============================================================================
# EFFECT SIZES (Cohen's d)
# ============================================================================

print("\n" + "=" * 70)
print("EFFECT SIZES (Cohen's d)")
print("=" * 70)

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Effect size for presentation method
d_presentation = cohens_d(headphone_acc, speaker_acc)
print(f"\nPresentation Method (Headphone vs Speaker):")
print(f"  Cohen's d = {d_presentation:.3f}")
if abs(d_presentation) < 0.2:
    print(f"  Interpretation: Small effect")
elif abs(d_presentation) < 0.5:
    print(f"  Interpretation: Small to medium effect")
elif abs(d_presentation) < 0.8:
    print(f"  Interpretation: Medium to large effect")
else:
    print(f"  Interpretation: Large effect")

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================

# Save condition statistics
condition_stats_file = script_dir / 'condition_statistics.csv'
condition_stats.to_csv(condition_stats_file, index=False)
print(f"\n" + "=" * 70)
print(f"Condition statistics saved to: {condition_stats_file}")

# Save combined trial data
combined_data_file = script_dir / 'combined_trial_data.csv'
combined_data.to_csv(combined_data_file, index=False)
print(f"Combined trial data saved to: {combined_data_file}")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)



