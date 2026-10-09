import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from scipy import stats

# Hardcoded input folder
DATA_FOLDER = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\results\data_28_09_26"

def analyze_participant_trials(trials_df, participant_id):
    """
    Analyze trials for a single participant.
    Returns a dictionary with counts and percentages for different groupings.
    """
    results = {}

    # 1. Count 'up' and 'down' responses for each presentation_type
    presentation_type_counts = trials_df.groupby('presentation_type')['response'].value_counts().unstack(fill_value=0)
    presentation_type_pct = presentation_type_counts.div(presentation_type_counts.sum(axis=1), axis=0) * 100

    # Flatten and store
    for ptype in presentation_type_pct.index:
        for response in presentation_type_pct.columns:
            key = f"presentation_type_{ptype}_response_{response}"
            results[key] = presentation_type_pct.loc[ptype, response]

    # 2. Count 'up' and 'down' responses for each stimulus_category
    stimulus_category_counts = trials_df.groupby('stimulus_category')['response'].value_counts().unstack(fill_value=0)
    stimulus_category_pct = stimulus_category_counts.div(stimulus_category_counts.sum(axis=1), axis=0) * 100

    for scat in stimulus_category_pct.index:
        for response in stimulus_category_pct.columns:
            key = f"stimulus_category_{scat}_response_{response}"
            results[key] = stimulus_category_pct.loc[scat, response]

    # 3. Count 'up' and 'down' responses for each presentation_type + stimulus_category combination
    combination_counts = trials_df.groupby(['presentation_type', 'stimulus_category'])['response'].value_counts().unstack(fill_value=0)
    combination_pct = combination_counts.div(combination_counts.sum(axis=1), axis=0) * 100

    for (ptype, scat) in combination_pct.index:
        for response in combination_pct.columns:
            key = f"presentation_type_{ptype}_stimulus_category_{scat}_response_{response}"
            results[key] = combination_pct.loc[(ptype, scat), response]

    return results


def prepare_anova_data(all_trial_files):
    """
    Prepare data for two-way ANOVA.
    Returns a DataFrame with one row per participant per condition,
    with the percentage of 'up' responses as the dependent variable.
    """
    anova_data = []

    for trial_file in all_trial_files:
        participant_id = trial_file.stem.split("_")[0]

        try:
            trials_df = pd.read_csv(trial_file)

            # Group by presentation_type and stimulus_category
            # and calculate percentage of 'up' responses
            grouped = trials_df.groupby(['presentation_type', 'stimulus_category']).apply(
                lambda x: (x['response'] == 'up').sum() / len(x) * 100,
                include_groups=False
            ).reset_index()
            grouped.columns = ['presentation_type', 'stimulus_category', 'up_response_pct']
            grouped['participant_id'] = participant_id

            anova_data.append(grouped)
        except Exception as e:
            print(f"Error preparing ANOVA data for participant {participant_id}: {e}")

    return pd.concat(anova_data, ignore_index=True) if anova_data else pd.DataFrame()


def run_two_way_anova(anova_df, output_path):
    """
    Run a 3x3 two-way ANOVA with presentation_type and stimulus_category as factors,
    and percentage of 'up' responses as the dependent variable.
    """
    if anova_df.empty:
        print("No data available for ANOVA")
        return

    # Fit the model: up_response_pct ~ presentation_type + stimulus_category + interaction
    model = ols('up_response_pct ~ C(presentation_type) + C(stimulus_category) + C(presentation_type):C(stimulus_category)',
                 data=anova_df).fit()

    # Generate ANOVA table
    anova_table = anova_lm(model, typ=2)

    # Print results
    print("\n" + "="*80)
    print("TWO-WAY ANOVA RESULTS (3x3)")
    print("Dependent Variable: Percentage of 'Up' Responses")
    print("="*80)
    print(f"\n{anova_table}\n")
    print("="*80)
    print(f"Model R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print("="*80 + "\n")

    # Save ANOVA table to CSV
    anova_output_file = output_path / "anova_results.csv"
    anova_table.to_csv(anova_output_file)
    print(f"ANOVA results saved to {anova_output_file}")

    # Save the prepared data used for ANOVA
    anova_data_file = output_path / "anova_data.csv"
    anova_df.to_csv(anova_data_file, index=False)
    print(f"ANOVA data saved to {anova_data_file}")


def check_normality(anova_df, output_path):
    """
    Check normality of the dependent variable (up_response_pct) using
    visual methods and statistical tests.
    """
    data = anova_df['up_response_pct'].dropna()

    # Statistical tests
    shapiro_stat, shapiro_p = stats.shapiro(data)
    anderson_result = stats.anderson(data)

    print("\n" + "="*80)
    print("NORMALITY TESTS")
    print("="*80)
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  p-value: {shapiro_p:.4f}")
    print(f"  Result: {'NORMAL' if shapiro_p > 0.05 else 'NOT NORMAL'} (α=0.05)\n")

    print(f"Anderson-Darling Test:")
    print(f"  Statistic: {anderson_result.statistic:.4f}")
    print(f"  Critical values: {anderson_result.critical_values}")
    print(f"  Significance levels: {anderson_result.significance_level}%\n")
    print("="*80 + "\n")

    # Visual checks
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Histogram with normal curve overlay
    axes[1].hist(data, bins=15, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal distribution')
    axes[1].set_title("Histogram with Normal Curve", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Up Response %")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Box plot
    axes[2].boxplot(data, vert=True)
    axes[2].set_title("Box Plot", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Up Response %")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    normality_plot_file = output_path / "normality_checks.png"
    plt.savefig(normality_plot_file, dpi=300, bbox_inches='tight')
    print(f"Normality plots saved to {normality_plot_file}\n")
    plt.close()



def main():
    # Initialize list to store all participant results
    all_participant_results = []

    # Get all trial CSV files
    data_path = Path(DATA_FOLDER)
    trial_files = sorted(data_path.glob("*_trials.csv"))

    if not trial_files:
        print(f"No trial files found in {DATA_FOLDER}")
        return

    # Process each participant
    for trial_file in trial_files:
        # Extract participant ID from filename (e.g., "024_trials.csv" -> "024")
        participant_id = trial_file.stem.split("_")[0]

        try:
            # Read trials data
            trials_df = pd.read_csv(trial_file)

            # Analyze this participant
            participant_results = analyze_participant_trials(trials_df, participant_id)
            participant_results['participant_id'] = participant_id

            all_participant_results.append(participant_results)
            print(f"Processed participant {participant_id}")

        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")

    if not all_participant_results:
        print("No participant data was successfully processed")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(all_participant_results)

    # Calculate averages and standard deviations across all participants
    # Get all columns except participant_id
    metric_columns = [col for col in results_df.columns if col != 'participant_id']

    averages = {}
    std_devs = {}

    for col in metric_columns:
        averages[col] = results_df[col].mean()
        std_devs[col] = results_df[col].std()

    # Create summary row for averages and std devs
    avg_row = {'participant_id': 'AVERAGE'} | averages
    std_row = {'participant_id': 'STD_DEV'} | std_devs

    # Combine individual results with averages and std devs
    final_results = pd.concat([
        results_df,
        pd.DataFrame([avg_row]),
        pd.DataFrame([std_row])
    ], ignore_index=True)

    # Save to results.csv
    output_file = data_path / "results.csv"
    final_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"\nProcessed {len(all_participant_results)} participants")

    # Run two-way ANOVA
    anova_df = prepare_anova_data(trial_files)
    run_two_way_anova(anova_df, data_path)

    # Check normality assumptions
    check_normality(anova_df, data_path)




if __name__ == "__main__":
    main()
