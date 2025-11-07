import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def create_oversmoothed_plot(config):
    """
    Loads tool data, normalizes and shifts it to its minimum, applies a
    circular Savitzky-Golay filter to find the underlying shape, and
    generates a publication-quality plot.

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    # --- Setup ---
    # Unpack configuration for easier access
    input_file = config['INPUT_FILE']
    output_dir = config['OUTPUT_DIR']
    window_size = config['SAVGOL_WINDOW']
    poly_order = config['SAVGOL_POLY_ORDER']

    # Set plot style and create output directory
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 16})
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and Process Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    # Ensure the data is sorted by degree for correct processing
    df = df.sort_values(by='Degree').reset_index(drop=True)
    
    # Normalize the 'Sum of Pixels' data to a 0-1 range
    df['Normalized Area'] = (df['Sum of Pixels'] - df['Sum of Pixels'].min()) / (df['Sum of Pixels'].max() - df['Sum of Pixels'].min())
    
    # --- Shift the Data to its Minimum ---
    # Find the index of the minimum value in the normalized data
    min_index = df['Normalized Area'].idxmin()
    
    # Reorder the DataFrame so the row with the minimum value is first
    df_shifted = pd.concat([df.loc[min_index:], df.loc[:min_index - 1]]).reset_index(drop=True)

    # Adjust the 'Degree' column so the new sequence starts at 0
    df_shifted['Degree'] = df_shifted['Degree'] - df_shifted.iloc[0]['Degree']
    # Wrap around degrees that become negative
    df_shifted.loc[df_shifted['Degree'] < 0, 'Degree'] += 360
    
    # --- Apply Circular Savitzky-Golay Filter ---
    # The 'mode="wrap"' argument handles the circular smoothing
    y_smoothed = savgol_filter(df_shifted['Normalized Area'], window_length=window_size, polyorder=poly_order, mode='wrap')

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. Plot the original, normalized, and shifted data as light gray dots
    ax.plot(df_shifted['Degree'], df_shifted['Normalized Area'], marker='.', linestyle='none', color='lightgray', label='Processed Raw Data')
    
    # 2. Plot the heavily smoothed Sav-Gol curve
    ax.plot(df_shifted['Degree'], y_smoothed, linestyle='-', color='#c44e52', label=f'Over-smoothed (Sav-Gol, w={window_size})')
    
    # --- Formatting ---
    ax.set_xlabel("Shifted Angle [°]", fontsize=14)
    ax.set_ylabel("Normalized Area", fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    
    # Set a fixed position for the legend at the top center, outside the plot area
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12, frameon=False)
    
    # --- Save the Figure ---
    # Generate a descriptive output filename based on the input file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{base_name}_oversmoothed_analysis.svg"
    plot_path = os.path.join(output_dir, output_filename)
    
    # Use bbox_inches='tight' to ensure the legend is not cut off
    fig.savefig(plot_path, format='svg', bbox_inches='tight')
    print(f"Plot saved to '{plot_path}'")
    plt.close(fig)


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        # --- Input File ---
        # Change this to the CSV file you want to analyze
        'INPUT_FILE': '../modified_chamfer_1edge_fractured.csv',
        
        # --- Output Directory ---
        'OUTPUT_DIR': 'oversmoothed_results',
        
        # --- Savitzky-Golay Filter Settings ---
        'SAVGOL_WINDOW': 71, # Must be an odd number
        'SAVGOL_POLY_ORDER': 2
    }
    
    # --- Run the Analysis for the first file ---
    create_oversmoothed_plot(config)
    
    # # --- Run for a second file to confirm consistent legend placement ---
    # config['INPUT_FILE'] = 'chamfer_processed.csv'
    # create_oversmoothed_plot(config)
