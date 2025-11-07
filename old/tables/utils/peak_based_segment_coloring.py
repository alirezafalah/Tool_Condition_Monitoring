import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import itertools

def plot_colored_segments(config):
    """
    Loads tool data, preprocesses it (normalizes and shifts), finds peaks on
    an over-smoothed version, and generates a plot with colored vertical bands
    representing the segments between the peaks.

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    # --- Setup ---
    input_file = config['INPUT_FILE']
    output_dir = config['OUTPUT_DIR']
    window_size = config['SAVGOL_WINDOW']
    poly_order = config['SAVGOL_POLY_ORDER']
    peak_distance = config['PEAK_DISTANCE']
    peak_prominence = config['PEAK_PROMINENCE']

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 16})
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and Process Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    df = df.sort_values(by='Degree').reset_index(drop=True)
    df['Normalized Area'] = (df['Sum of Pixels'] - df['Sum of Pixels'].min()) / (df['Sum of Pixels'].max() - df['Sum of Pixels'].min())
    
    # --- Shift the Data to its Minimum ---
    min_index = df['Normalized Area'].idxmin()
    df_shifted = pd.concat([df.loc[min_index:], df.loc[:min_index - 1]]).reset_index(drop=True)
    df_shifted['Degree'] = df_shifted['Degree'] - df_shifted.iloc[0]['Degree']
    df_shifted.loc[df_shifted['Degree'] < 0, 'Degree'] += 360
    
    # --- Apply Savitzky-Golay Filter ---
    y_smoothed = savgol_filter(df_shifted['Normalized Area'], window_length=window_size, polyorder=poly_order, mode='wrap')

    # --- Find Peaks on the Smoothed Data ---
    peaks, _ = find_peaks(y_smoothed, distance=peak_distance, prominence=peak_prominence)
    print(f"Detected {len(peaks)} peaks for '{os.path.basename(input_file)}'.")
    if len(peaks) > 0:
        # Print the exact degree values of the detected peaks
        peak_degrees_print = np.round(df_shifted['Degree'][peaks].values, 2)
        print(f"Peak positions (degrees): {peak_degrees_print}")


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Color Segments Between Peaks (Corrected Wrap-Around Logic) ---
    if len(peaks) > 0:
        # The boundaries of the segments are the peaks themselves
        peak_degrees = df_shifted['Degree'][peaks].values
        num_peaks = len(peaks)
        
        # Define a list of colors for the bands
        colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 'cyan'])

        for i in range(num_peaks):
            color = next(colors)
            start_degree = peak_degrees[i]
            # The next peak, wrapping around using the modulo operator
            end_degree = peak_degrees[(i + 1) % num_peaks]

            if start_degree > end_degree:
                # This is the wrap-around segment. It needs two colored bands.
                # 1. From the last peak to the end of the graph
                ax.axvspan(start_degree, 360, color=color, alpha=0.25)
                # 2. From the beginning of the graph to the first peak
                ax.axvspan(0, end_degree, color=color, alpha=0.25)
            else:
                # This is a standard segment within the 0-360 range
                ax.axvspan(start_degree, end_degree, color=color, alpha=0.25)

    # Plot the smoothed data line on top of the colored bands
    ax.plot(df_shifted['Degree'], y_smoothed, linestyle='-', color='black', label='Smoothed Signal')
    
    # --- Formatting ---
    ax.set_xlabel("Shifted Angle [°]", fontsize=14)
    ax.set_ylabel("Normalized Area", fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    ax.set_xticks(np.arange(0, 361, 60))
    ax.legend(loc='upper right', fontsize=12)
    
    # --- Save the Figure ---
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{base_name}_colored_segments.svg"
    plot_path = os.path.join(output_dir, output_filename)
    
    fig.savefig(plot_path, format='svg', bbox_inches='tight')
    print(f"Plot saved to '{plot_path}'")
    plt.close(fig)


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'INPUT_FILE': '../modified_chamfer_1edge_fractured.csv',
        'OUTPUT_DIR': 'chamfer_fractured_colored_segments',
        'SAVGOL_WINDOW': 71,
        'SAVGOL_POLY_ORDER': 2,
        'PEAK_DISTANCE': 30,
        'PEAK_PROMINENCE': 0.2
    }
    
    # --- Run the Analysis ---
    plot_colored_segments(config)
    
