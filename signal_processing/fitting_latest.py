import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools

def sinusoidal(x, amplitude, frequency, phase_shift, vertical_offset):
    """Defines a sinusoidal function for curve fitting."""
    return amplitude * np.sin(frequency * np.radians(x) + phase_shift) + vertical_offset

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared value for a given fit."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Handle the case where ss_tot is zero to avoid division by zero
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

def fit_and_plot_segments(config):
    """
    Loads processed tool data, splits it into segments, fits a sinusoidal
    curve to each segment, and generates a publication-quality plot with
    all segments overlaid for comparison.

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    # --- Setup ---
    input_file = config['INPUT_FILE']
    output_dir = config['OUTPUT_DIR']
    num_segments = config['NUM_SEGMENTS']

    # Set plot style and create output directory
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 16})
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    # --- Segment Data ---
    segment_size = len(df) // num_segments
    segments = [df.iloc[i * segment_size: (i + 1) * segment_size].reset_index(drop=True)
                for i in range(num_segments)]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(12, 7))
    # Use a new, more distinct list of colors that cycles if needed
    colors = itertools.cycle(["blue", "green", "red", "purple", "orange", "cyan", "magenta", "brown"])
    results = []

    # --- Fit and Plot Each Segment ---
    for i, segment in enumerate(segments):
        # Shift each segment's degrees to start from 0 for independent fitting
        x_data = segment['Degree'].values - segment['Degree'].values[0]
        y_data = segment['Sum of Pixels'].values

        # Define an initial guess for the curve fitting parameters
        initial_guess = [
            np.ptp(y_data) / 2,  # Amplitude
            num_segments,        # Frequency (should be close to the number of segments)
            0,                   # Phase Shift
            np.mean(y_data)      # Vertical Offset
        ]

        try:
            # Fit the sinusoidal function to the segment data
            popt, _ = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
            r2 = calculate_r2(y_data, sinusoidal(x_data, *popt))
        except RuntimeError:
            print(f"Warning: Curve fit failed for Segment {i+1}. Skipping.")
            popt = [0, 0, 0, 0]
            r2 = 0.0

        results.append({
            'Segment': i + 1, 'Amplitude': popt[0], 'Frequency': popt[1],
            'Phase Shift': popt[2], 'Vertical Offset': popt[3], 'R²': r2
        })

        color = next(colors)
        # Plot the original data points for the segment, now starting from 0
        ax.plot(x_data, y_data, marker='.', linestyle='none', color=color, alpha=0.4)
        
        # Plot the fitted sinusoidal curve over the relative degree range
        x_fit = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit = sinusoidal(x_fit, *popt)
        
        ax.plot(x_fit, y_fit, color=color, linestyle='--',
                label=f'Segment {i+1} Fit (R²={r2:.3f})')

    # --- Final Plot Formatting ---
    ax.set_xlabel("Relative Angle within Segment [°]", fontsize=14)
    ax.set_ylabel("Normalized ROI Area", fontsize=14)
    # Adjust x-axis limit to the length of a single segment
    ax.set_xlim(0, 360 / num_segments)
    
    # Set a fixed position for the legend at the top center
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(num_segments, 4), frameon=False, fontsize=12)

    # --- Save Figure and Results ---
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    plot_filename = f"{base_name}_overlaid_sinusoidal_fit.svg"
    results_filename = f"{base_name}_fit_parameters.csv"
    
    plot_path = os.path.join(output_dir, plot_filename)
    results_path = os.path.join(output_dir, results_filename)

    fig.savefig(plot_path, format='svg', bbox_inches='tight')
    print(f"Plot saved to '{plot_path}'")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"Fit parameters saved to '{results_path}'")
    print("\n--- Fit Results ---")
    print(results_df.to_string(index=False))
    
    plt.close(fig)


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        # Path to the PROCESSED (shifted and scaled) data file
        'INPUT_FILE': '../tables/processed_data_drill_intact/processed_tool_data.csv',
        
        # Directory where the output plot and parameters will be saved
        'OUTPUT_DIR': 'sinusoidal_fit_results',
        
        # The number of segments to fit. Should match the previous script.
        'NUM_SEGMENTS': 2
    }
    
    # --- Run the Analysis ---
    fit_and_plot_segments(config)
