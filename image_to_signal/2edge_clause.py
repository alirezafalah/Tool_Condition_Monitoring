import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_signal(df: pd.DataFrame, angle_col: str, value_col: str) -> pd.DataFrame:
    """Performs scaling and shifting on the tool signal data."""
    print("Step 1: Scaling data to the [0, 1] range...")
    min_val = df[value_col].min()
    max_val = df[value_col].max()
    if max_val - min_val == 0:
        df_scaled = df.copy()
        df_scaled[value_col] = 0.5
    else:
        df_scaled = df.copy()
        df_scaled[value_col] = (df[value_col] - min_val) / (max_val - min_val)
    
    print("Step 2: Shifting signal so that the minimum value is at 0 degrees...")
    min_index = df_scaled[value_col].idxmin()
    df_shifted = pd.concat([df_scaled.loc[min_index:], df_scaled.loc[:min_index - 1]]).reset_index(drop=True)

    start_degree = df_shifted.iloc[0][angle_col]
    df_shifted[angle_col] = df_shifted[angle_col] - start_degree
    df_shifted.loc[df_shifted[angle_col] < 0, angle_col] += 360
    
    df_shifted = df_shifted.sort_values(by=angle_col).reset_index(drop=True)

    print("Preprocessing complete.\n")
    return df_shifted


def calculate_max_deviation_score(signal_1: np.ndarray, signal_2: np.ndarray) -> tuple:
    """
    Calculates the Maximum Deviation Ratio score.
    Returns: A tuple containing (score, difference_signal, std_dev_diff, max_abs_diff).
    """
    print("Step 3: Calculating Maximum Deviation Ratio...")
    difference = signal_1 - signal_2
    
    if np.all(difference == 0):
        print("Analysis complete: Signal halves are identical.")
        return 0.0, difference, 0.0, 0.0

    std_dev_diff = np.std(difference)
    max_abs_diff = np.max(np.abs(difference))
    
    score = np.inf if std_dev_diff == 0 else max_abs_diff / std_dev_diff
        
    print("Analysis complete.")
    return score, difference, std_dev_diff, max_abs_diff

def create_deviation_plot(angles: np.ndarray, difference: np.ndarray, std_dev: float, max_dev: float, score: float, save_path: str, should_save: bool):
    """
    Creates and optionally saves a single plot of the difference signal.
    """
    print("Step 4: Generating deviation plot...")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the difference signal
    ax.plot(angles, difference, color='green', label='Difference Signal', zorder=2)
    ax.axhline(y=0, color='black', linestyle='-')
    
    # Plot standard deviation bands
    ax.fill_between(angles, -1*std_dev, 1*std_dev, color='blue', alpha=0.3, label='1σ Noise Band')
    ax.fill_between(angles, -3*std_dev, 3*std_dev, color='blue', alpha=0.1, label='3σ Noise Band')
    
    # Highlight the maximum deviation point
    max_dev_idx = np.argmax(np.abs(difference))
    ax.plot(angles[max_dev_idx], difference[max_dev_idx], 'rx', markersize=12, label=f'Max Deviation ({max_dev:.3f})')
    
    ax.set_title(f'Difference Signal Analysis (MDR Score: {score:.2f})')
    ax.set_xlabel('Angle (Degrees)')
    ax.set_ylabel('Difference')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if should_save:
        # Save the plot in EPS format for LaTeX
        plt.savefig(save_path, format='svg', dpi=300)
        print(f"Plot saved to: {save_path}")
        
    plt.show()

# ==============================================================================
# --- MAIN SCRIPT ---
# ==============================================================================
if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    # Put the path to the CSV you want to analyze here
    csv_file_path = "data/tool064gain10paperBG_area_vs_angle.csv" # Example for Healthy tool
    # csv_file_path = "data/tool074gain10paperBG_area_vs_angle.csv" # Example for fractured tool
    
    angle_column_name = "Angle (Degrees)"
    value_column_name = "Smoothed ROI Area"
    center = 180.0
    
    VISUALIZE_PLOT = True
    SAVE_PLOT = True
    # The output filename will be based on the input CSV name
    input_filename = csv_file_path.split('/')[-1].split('.')[0]
    PLOT_OUTPUT_PATH = f"{input_filename}_deviation_plot.svg"

    # --- SCRIPT EXECUTION ---
    try:
        print(f"Loading data from: {csv_file_path}")
        tool_df = pd.read_csv(csv_file_path)

        processed_df = preprocess_signal(tool_df, angle_column_name, value_column_name)
        
        half_1_df = processed_df[processed_df[angle_column_name] < center]
        half_2_df = processed_df[processed_df[angle_column_name] >= center]
        
        half_1_angles = half_1_df[angle_column_name].values
        half_1_values = half_1_df[value_column_name].values
        
        half_2_angles_shifted = half_2_df[angle_column_name].values - center
        half_2_values = half_2_df[value_column_name].values
        
        half_2_values_aligned = np.interp(half_1_angles, half_2_angles_shifted, half_2_values)
        
        deviation_score, diff_signal, std_diff, max_diff = calculate_max_deviation_score(half_1_values, half_2_values_aligned)
        
        print("\n" + "="*50)
        print(f"✅ FINAL DEVIATION SCORE: {deviation_score:.4f}")
        print("="*50)
        print("(Low score is GOOD, High score is BAD)\n")
        
        if VISUALIZE_PLOT:
            create_deviation_plot(
                half_1_angles,
                diff_signal,
                std_diff,
                max_diff,
                deviation_score,
                PLOT_OUTPUT_PATH,
                SAVE_PLOT
            )

    except FileNotFoundError:
        print(f"\n❌ ERROR: The file was not found at the specified path.")
        print(f"Please make sure the path is correct: '{csv_file_path}'")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")