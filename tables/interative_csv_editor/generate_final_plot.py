import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import os

def generate_final_plot():
    """
    Reads a cleaned CSV data file, optionally applies a moving average,
    and generates a final, high-quality plot.
    """
    print("--- Generating Final Plot from Cleaned CSV ---")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # CONTROL PANEL
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 1. Path to your interactively fixed CSV file
    INPUT_CSV = 'fixed_interactive_data.csv'

    # 2. Path to save the final plot image
    OUTPUT_PLOT = 'final_plot_from_interactive_data.svg'

    # 3. Optional: Apply a moving average to the cleaned data
    APPLY_MOVING_AVERAGE = True
    MOVING_AVERAGE_WINDOW = 5
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(INPUT_CSV)
        # Get column names dynamically
        x_col = df.columns[0]
        y_col = df.columns[1]
        print(f"Successfully loaded data from '{INPUT_CSV}'")
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{INPUT_CSV}'. Please check the path.")
        return
    except IndexError:
        print("Error: CSV file must have at least two columns.")
        return

    # --- 2. Apply Wrap-Around Moving Average (Optional) ---
    raw_data_col = y_col
    processed_data_col = y_col # Default to the raw data column

    if APPLY_MOVING_AVERAGE:
        print(f"Applying moving average with window size {MOVING_AVERAGE_WINDOW}...")
        weights = np.ones(MOVING_AVERAGE_WINDOW) / MOVING_AVERAGE_WINDOW
        smoothed_data = convolve1d(df[raw_data_col], weights=weights, mode='wrap')
        
        # Add the smoothed data to the DataFrame
        df['Smoothed ROI Area'] = smoothed_data
        processed_data_col = 'Smoothed ROI Area' # Update target column for plotting

    # --- 3. Plot Data ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # If smoothing was applied, show the manually cleaned data as a scatter plot
    if APPLY_MOVING_AVERAGE:
        ax.scatter(df[x_col], df[raw_data_col], color='lightgray', s=30, label='Raw Data')

    # Plot the main data (either cleaned or smoothed)
    ax.plot(df[x_col], df[processed_data_col], marker='.', linestyle='-', markersize=8, label='Smoothed (Moving Average)')
    
    ax.set_title('Tool ROI Area vs. Rotation Angle', fontsize=18, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    plt.tight_layout()
    
    # --- 4. Save and Show Plot ---
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_PLOT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(OUTPUT_PLOT, format='svg', dpi=300)
    print(f"Plot successfully saved to '{OUTPUT_PLOT}'")
    plt.show()

if __name__ == "__main__":
    generate_final_plot()
