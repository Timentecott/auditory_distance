import pandas as pd
import glob
import os
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# Define directory path
pilot_results_dir = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\results\pilot_results"

# Create a list of all CSV files ending with _results.csv
csv_files = glob.glob(os.path.join(pilot_results_dir, "*_results.csv"))

# Initialize a list to store results for all participants
results_list = []

# Process each file
for file_path in csv_files:
    # Extract participant name from file name (remove _results.csv suffix)
    file_name = os.path.basename(file_path)
    participant_name = file_name.replace("_results.csv", "")

    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check for required columns
    required_columns = ['correct', 'cue_to_dot_isi', 'validity_condition', 'response_time']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"WARNING: {participant_name} is missing columns: {missing_columns}")
        print(f"Available columns: {list(data.columns)}")
        continue

    # Normalize the 'correct' column to boolean (handle TRUE/True and FALSE/False strings)
    if data['correct'].dtype == 'object':
        data['correct'] = data['correct'].astype(str).str.lower() == 'true'

    # Normalize the 'validity_condition' column to have consistent capitalization
    data['validity_condition'] = data['validity_condition'].astype(str).str.capitalize()

    # Filter out incorrect trials (where correct == False)
    data_valid = data[data['correct'] == True].copy()

    # Convert cue_to_dot_isi to milliseconds for easier naming (0.2->200, 0.275->275, 0.35->350)
    data_valid['isi_ms'] = (data_valid['cue_to_dot_isi'] * 1000).astype(int)

    # Create a dictionary to store results for this participant
    participant_results = {'Participant': participant_name}

    # Calculate averages and validity effects for each ISI condition
    for isi_ms in [200, 275, 350]:
        # Filter data for this ISI
        isi_data = data_valid[data_valid['isi_ms'] == isi_ms]

        if len(isi_data) > 0:
            # Calculate average response times and SDs for valid and invalid conditions
            valid_trials = isi_data[isi_data['validity_condition'] == 'Valid']
            invalid_trials = isi_data[isi_data['validity_condition'] == 'Invalid']

            valid_rt = valid_trials['response_time'].mean() if len(valid_trials) > 0 else None
            valid_sd = valid_trials['response_time'].std() if len(valid_trials) > 0 else None
            invalid_rt = invalid_trials['response_time'].mean() if len(invalid_trials) > 0 else None
            invalid_sd = invalid_trials['response_time'].std() if len(invalid_trials) > 0 else None

            # Calculate validity effect (valid - invalid) if both conditions exist
            validity_effect = valid_rt - invalid_rt if (valid_rt is not None and invalid_rt is not None) else None

            # Count valid trials for this condition
            valid_trial_count = len(valid_trials)
            invalid_trial_count = len(invalid_trials)

            # Store results
            participant_results[f'avg_valid_{isi_ms}'] = valid_rt
            participant_results[f'sd_valid_{isi_ms}'] = valid_sd
            participant_results[f'avg_invalid_{isi_ms}'] = invalid_rt
            participant_results[f'sd_invalid_{isi_ms}'] = invalid_sd
            participant_results[f'validity_effect_{isi_ms}'] = validity_effect
            participant_results[f'valid_trial_count_{isi_ms}'] = valid_trial_count
            participant_results[f'invalid_trial_count_{isi_ms}'] = invalid_trial_count

    # Add this participant's results to the list
    results_list.append(participant_results)

# Create a DataFrame from all results
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
output_path = os.path.join(pilot_results_dir, "pilot_analysis_results.csv")
results_df.to_csv(output_path, index=False)

print(f"Analysis complete! Results saved to: {output_path}")
print(f"\nSummary of processed participants: {len(results_df)}")
print(results_df)

# ============================================================================
# WITHIN-SUBJECTS 2x3 ANOVA (Validity x ISI)
# ============================================================================
print("\n" + "="*80)
print("2 (Valid vs Invalid) x 3 (200, 275, 350 ms ISI) WITHIN-SUBJECTS ANOVA")
print("="*80)

# Prepare data for ANOVA: reshape to long format
long_data = []
for _, row in results_df.iterrows():
    participant = row['Participant']
    for isi_ms in [200, 275, 350]:
        for validity in ['Valid', 'Invalid']:
            validity_lower = validity.lower()
            rt_col = f'avg_{validity_lower}_{isi_ms}'
            if rt_col in results_df.columns:
                long_data.append({
                    'Participant': participant,
                    'Validity': validity,
                    'ISI_ms': isi_ms,
                    'RT': row[rt_col]
                })

anova_df = pd.DataFrame(long_data)

# Pivot table: participants in rows, condition combinations in columns
pivot_df = anova_df.pivot_table(
    index='Participant',
    columns=['Validity', 'ISI_ms'],
    values='RT'
)

print("\nAggregated Data (one row per participant):")
print(pivot_df)

# Group by validity and ISI to get condition means
condition_means = anova_df.groupby(['Validity', 'ISI_ms'])['RT'].agg(['mean', 'count']).reset_index()
print(f"\nCondition Means:")
print(condition_means)

# Conduct repeated measures ANOVA using statsmodels
aov = AnovaRM(anova_df, 'RT', 'Participant', within=['Validity', 'ISI_ms'])
anova_result = aov.fit()

print("\n" + "="*80)
print("ANOVA TABLE")
print("="*80)
print(anova_result)

# Summary statistics by condition
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS BY CONDITION")
print("="*80)
summary_stats = anova_df.groupby(['Validity', 'ISI_ms'])['RT'].agg(['count', 'mean', 'std', 'sem'])
print(summary_stats)

print("\n" + "="*80)
