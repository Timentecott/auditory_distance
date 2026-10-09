import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Hardcoded input folder
DATA_FOLDER = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_2\results\experiment_end_sept"

def analyze_participant_results(results_df, participant_id):
    """
    Analyze results for a single participant.
    Returns a dictionary with validity effects and RT metrics.
    """
    results = {'participant_id': participant_id}

    # Exclude incorrect trials
    results_df = results_df[results_df['correct'] == True]
    n_trials_before = len(results_df)

    # Exclude RTs beyond mean + 2 SD (only trim slow RTs)
    mean_rt = results_df['response_time'].mean()
    std_rt = results_df['response_time'].std()
    upper_bound = mean_rt + 2 * std_rt
    results_df = results_df[results_df['response_time'] <= upper_bound]

    n_trials_after = len(results_df)
    percent_removed = ((n_trials_before - n_trials_after) / n_trials_before * 100) if n_trials_before > 0 else 0
    print(f"{participant_id}: {n_trials_before} -> {n_trials_after} trials ({percent_removed:.1f}% removed)")

    # Return NaN values if no data remains after cleaning
    if len(results_df) == 0:
        results['overall_validity_effect'] = np.nan
        results['loudspeaker_validity_effect'] = np.nan
        results['in_situ_validity_effect'] = np.nan
        results['ex_situ_validity_effect'] = np.nan
        results['mean_rt_near'] = np.nan
        results['mean_rt_far'] = np.nan
        return results

    # Overall validity effect
    invalid_rt_overall = results_df[results_df['validity_condition'] == 'Invalid']['response_time'].mean()
    valid_rt_overall = results_df[results_df['validity_condition'] == 'Valid']['response_time'].mean()
    results['overall_validity_effect'] = invalid_rt_overall - valid_rt_overall

    # Loudspeaker condition validity effect
    loudspeaker_df = results_df[results_df['presentation_type'] == 'loudspeaker']
    if len(loudspeaker_df) > 0:
        invalid_rt_loudspeaker = loudspeaker_df[loudspeaker_df['validity_condition'] == 'Invalid']['response_time'].mean()
        valid_rt_loudspeaker = loudspeaker_df[loudspeaker_df['validity_condition'] == 'Valid']['response_time'].mean()
        results['loudspeaker_validity_effect'] = invalid_rt_loudspeaker - valid_rt_loudspeaker
    else:
        results['loudspeaker_validity_effect'] = np.nan

    # In_situ condition validity effect
    insitu_df = results_df[results_df['presentation_type'] == 'in_situ']
    if len(insitu_df) > 0:
        invalid_rt_insitu = insitu_df[insitu_df['validity_condition'] == 'Invalid']['response_time'].mean()
        valid_rt_insitu = insitu_df[insitu_df['validity_condition'] == 'Valid']['response_time'].mean()
        results['in_situ_validity_effect'] = invalid_rt_insitu - valid_rt_insitu
    else:
        results['in_situ_validity_effect'] = np.nan

    # Ex_situ condition validity effect
    exsitu_df = results_df[results_df['presentation_type'] == 'ex_situ']
    if len(exsitu_df) > 0:
        invalid_rt_exsitu = exsitu_df[exsitu_df['validity_condition'] == 'Invalid']['response_time'].mean()
        valid_rt_exsitu = exsitu_df[exsitu_df['validity_condition'] == 'Valid']['response_time'].mean()
        results['ex_situ_validity_effect'] = invalid_rt_exsitu - valid_rt_exsitu
    else:
        results['ex_situ_validity_effect'] = np.nan

    # Mean response time to dot location near
    near_rt = results_df[results_df['dot_location'] == 'near']['response_time'].mean()
    results['mean_rt_near'] = near_rt

    # Mean response time to dot location far
    far_rt = results_df[results_df['dot_location'] == 'far']['response_time'].mean()
    results['mean_rt_far'] = far_rt

    return results


def check_normality(results_df, output_path):
    """
    Check normality of dependent variables (validity effects and RTs) using
    statistical tests and visual methods.
    """
    # Variables to test
    variables_to_test = [
        'overall_validity_effect',
        'loudspeaker_validity_effect',
        'in_situ_validity_effect',
        'ex_situ_validity_effect',
        'mean_rt_near',
        'mean_rt_far'
    ]

    # Store results
    normality_results = []

    print("\n" + "="*80)
    print("NORMALITY TESTS - All Dependent Variables")
    print("="*80 + "\n")

    # Create figure for Q-Q plots and histograms
    fig, axes = plt.subplots(len(variables_to_test), 3, figsize=(18, 4*len(variables_to_test)))

    for idx, var in enumerate(variables_to_test):
        data = results_df[var].dropna()

        if len(data) < 3:
            print(f"Skipping {var}: insufficient data (n={len(data)})")
            continue

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Anderson-Darling test
        anderson_result = stats.anderson(data)

        # Store results
        normality_results.append({
            'Variable': var,
            'N': len(data),
            'Shapiro-Wilk_Statistic': shapiro_stat,
            'Shapiro-Wilk_p_value': shapiro_p,
            'Normal (α=0.05)': 'Yes' if shapiro_p > 0.05 else 'No',
            'Anderson-Darling_Statistic': anderson_result.statistic
        })

        # Print results
        print(f"{var.upper()}")
        print(f"  N: {len(data)}")
        print(f"  Shapiro-Wilk: W({len(data)}) = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
        print(f"  Result: {'NORMAL' if shapiro_p > 0.05 else 'NOT NORMAL'} (α=0.05)")
        print(f"  Anderson-Darling: A² = {anderson_result.statistic:.4f}\n")

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[idx, 0])
        axes[idx, 0].set_title(f"Q-Q Plot: {var}", fontsize=10, fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)

        # Histogram with normal curve overlay
        axes[idx, 1].hist(data, bins=12, density=True, alpha=0.7, edgecolor='black')
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        axes[idx, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal dist.')
        axes[idx, 1].set_title(f"Histogram: {var}", fontsize=10, fontweight='bold')
        axes[idx, 1].set_ylabel("Density")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(True, alpha=0.3)

        # Box plot
        axes[idx, 2].boxplot(data, vert=True)
        axes[idx, 2].set_title(f"Box Plot: {var}", fontsize=10, fontweight='bold')
        axes[idx, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    normality_plot_file = output_path / "normality_checks.png"
    plt.savefig(normality_plot_file, dpi=300, bbox_inches='tight')
    print("="*80)
    print(f"Normality plots saved to {normality_plot_file}\n")
    plt.close()

    # Save normality results to CSV
    normality_df = pd.DataFrame(normality_results)
    normality_output_file = output_path / "normality_results.csv"
    normality_df.to_csv(normality_output_file, index=False)
    print(f"Normality results summary saved to {normality_output_file}\n")


def prepare_anova_data_validity_effect(results_df):
    """
    Prepare data for one-way ANOVA comparing validity effects across presentation types.
    Returns a DataFrame with one row per participant per presentation type.
    """
    anova_data = []

    for _, row in results_df.iterrows():
        participant_id = row['participant_id']

        # Skip AVERAGE and STD_DEV rows
        if participant_id in ['AVERAGE', 'STD_DEV']:
            continue

        # loudspeaker
        if pd.notna(row['loudspeaker_validity_effect']):
            anova_data.append({
                'participant_id': participant_id,
                'presentation_type': 'loudspeaker',
                'validity_effect': row['loudspeaker_validity_effect']
            })

        # in_situ
        if pd.notna(row['in_situ_validity_effect']):
            anova_data.append({
                'participant_id': participant_id,
                'presentation_type': 'in_situ',
                'validity_effect': row['in_situ_validity_effect']
            })

        # ex_situ
        if pd.notna(row['ex_situ_validity_effect']):
            anova_data.append({
                'participant_id': participant_id,
                'presentation_type': 'ex_situ',
                'validity_effect': row['ex_situ_validity_effect']
            })

    return pd.DataFrame(anova_data)


def run_one_way_anova_validity_effect(anova_df, output_path):
    """
    Run a one-way ANOVA comparing validity effects across presentation types
    (loudspeaker, in_situ, ex_situ).
    """
    if anova_df.empty:
        print("No data available for ANOVA")
        return

    # Get number of participants
    n_participants = anova_df['participant_id'].nunique()
    n_total = len(anova_df)

    # Fit the model: validity_effect ~ presentation_type
    model = ols('validity_effect ~ C(presentation_type)', data=anova_df).fit()

    # Generate ANOVA table
    anova_table = anova_lm(model, typ=2)

    # Print results
    print("\n" + "="*80)
    print("ONE-WAY ANOVA: VALIDITY EFFECT BY PRESENTATION TYPE")
    print("="*80)
    print(f"Sample: {n_participants} participants, {n_total} total observations")
    print(f"Factor: Presentation Type (loudspeaker, in_situ, ex_situ)")
    print(f"Dependent Variable: Validity Effect (Invalid RT - Valid RT, in seconds)\n")
    print(f"{anova_table}\n")
    print("="*80)
    print(f"Model R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print("="*80 + "\n")

    # Descriptive statistics by presentation type
    print("Descriptive Statistics (in ms):")
    for ptype in sorted(anova_df['presentation_type'].unique()):
        data = anova_df[anova_df['presentation_type'] == ptype]['validity_effect'] * 1000
        print(f"  {ptype.upper():15} M = {data.mean():7.2f}, SD = {data.std():7.2f}, n = {len(data)}")
    print()

    # Save ANOVA table to CSV
    anova_output_file = output_path / "anova_validity_effect_by_presentation.csv"
    anova_table.to_csv(anova_output_file)
    print(f"ANOVA results saved to {anova_output_file}")

    # Save the prepared data used for ANOVA
    anova_data_file = output_path / "anova_data_validity_effect.csv"
    anova_df.to_csv(anova_data_file, index=False)
    print(f"ANOVA data saved to {anova_data_file}\n")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    presentation_types = sorted(anova_df['presentation_type'].unique())
    box_data = [anova_df[anova_df['presentation_type'] == pt]['validity_effect'].values * 1000 
                for pt in presentation_types]
    axes[0].boxplot(box_data, labels=presentation_types)
    axes[0].set_ylabel('Validity Effect (ms)', fontsize=11)
    axes[0].set_xlabel('Presentation Type', fontsize=11)
    axes[0].set_title('Validity Effect by Presentation Type', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Bar plot with error bars
    means = [anova_df[anova_df['presentation_type'] == pt]['validity_effect'].mean() * 1000 
             for pt in presentation_types]
    stds = [anova_df[anova_df['presentation_type'] == pt]['validity_effect'].std() * 1000 
            for pt in presentation_types]
    x_pos = np.arange(len(presentation_types))
    axes[1].bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Validity Effect (ms)', fontsize=11)
    axes[1].set_xlabel('Presentation Type', fontsize=11)
    axes[1].set_title('Mean Validity Effect by Presentation Type (±SD)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(presentation_types)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No effect')
    axes[1].legend()

    plt.tight_layout()
    plot_file = output_path / "anova_validity_effect_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_file}\n")
    plt.close()



def main():
    # Initialize list to store all participant results
    all_participant_results = []

    # Get all result CSV files
    data_path = Path(DATA_FOLDER)
    result_files = sorted(data_path.glob("*_results.csv"))

    if not result_files:
        print(f"No result files found in {DATA_FOLDER}")
        return

    # Process each participant
    for result_file in result_files:
        # Extract participant ID from filename (e.g., "002_results.csv" -> "002")
        participant_id = result_file.stem.split("_")[0]

        try:
            # Read results data
            results_df = pd.read_csv(result_file)

            # Analyze this participant
            participant_results = analyze_participant_results(results_df, participant_id)

            all_participant_results.append(participant_results)
            print(f"Processed participant {participant_id}")

        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")

    if not all_participant_results:
        print("No participant data was successfully processed")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(all_participant_results)

    # Calculate averages across all participants
    avg_row = {
        'participant_id': 'AVERAGE',
        'overall_validity_effect': results_df['overall_validity_effect'].mean(),
        'loudspeaker_validity_effect': results_df['loudspeaker_validity_effect'].mean(),
        'in_situ_validity_effect': results_df['in_situ_validity_effect'].mean(),
        'ex_situ_validity_effect': results_df['ex_situ_validity_effect'].mean(),
        'mean_rt_near': results_df['mean_rt_near'].mean(),
        'mean_rt_far': results_df['mean_rt_far'].mean()
    }

    # Calculate standard deviations across all participants (excluding NaN values)
    std_row = {
        'participant_id': 'STD_DEV',
        'overall_validity_effect': results_df['overall_validity_effect'].std(skipna=True),
        'loudspeaker_validity_effect': results_df['loudspeaker_validity_effect'].std(skipna=True),
        'in_situ_validity_effect': results_df['in_situ_validity_effect'].std(skipna=True),
        'ex_situ_validity_effect': results_df['ex_situ_validity_effect'].std(skipna=True),
        'mean_rt_near': results_df['mean_rt_near'].std(skipna=True),
        'mean_rt_far': results_df['mean_rt_far'].std(skipna=True)
    }

    # Combine individual results with averages and std devs
    final_results = pd.concat([
        results_df,
        pd.DataFrame([avg_row]),
        pd.DataFrame([std_row])
    ], ignore_index=True)

    # Save to results.csv
    output_file = data_path / "results_summary.csv"
    final_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"\nProcessed {len(all_participant_results)} participants")
    print(f"\nDiagnostics:")
    print(f"  Valid values in loudspeaker_validity_effect: {results_df['loudspeaker_validity_effect'].notna().sum()}")
    print(f"  Valid values in in_situ_validity_effect: {results_df['in_situ_validity_effect'].notna().sum()}")
    print(f"  Valid values in ex_situ_validity_effect: {results_df['ex_situ_validity_effect'].notna().sum()}")
    print(f"\nSummary (in ms):")
    print(f"Mean overall validity effect: {avg_row['overall_validity_effect']*1000:.2f} ± {std_row['overall_validity_effect']*1000:.2f} ms")
    print(f"Mean loudspeaker validity effect: {avg_row['loudspeaker_validity_effect']*1000:.2f} ± {std_row['loudspeaker_validity_effect']*1000:.2f} ms")
    print(f"Mean in_situ validity effect: {avg_row['in_situ_validity_effect']*1000:.2f} ± {std_row['in_situ_validity_effect']*1000:.2f} ms")
    print(f"Mean ex_situ validity effect: {avg_row['ex_situ_validity_effect']*1000:.2f} ± {std_row['ex_situ_validity_effect']*1000:.2f} ms")
    print(f"Mean RT to near: {avg_row['mean_rt_near']*1000:.2f} ± {std_row['mean_rt_near']*1000:.2f} ms")
    print(f"Mean RT to far: {avg_row['mean_rt_far']*1000:.2f} ± {std_row['mean_rt_far']*1000:.2f} ms")

    # Check normality of dependent variables
    check_normality(results_df, data_path)

    # Run one-way ANOVA for validity effect across presentation types
    anova_validity_df = prepare_anova_data_validity_effect(results_df)
    run_one_way_anova_validity_effect(anova_validity_df, data_path)




if __name__ == "__main__":
    main()


