import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

def circular_moving_average(data, window_size):
    """
    Calculates a circular moving average for a given 1D array of data.
    The "wrap-around" is handled by padding the array with elements
    from the other end before calculating the moving average.

    Args:
        data (np.array or pd.Series): The input data to be smoothed.
        window_size (int): The size of the moving average window. Must be an odd number.

    Returns:
        np.array: The smoothed data array.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number to ensure symmetry.")
    
    pad_size = (window_size - 1) // 2
    padded_data = np.pad(data, pad_width=pad_size, mode='wrap')
    weights = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, weights, mode='valid')
    return smoothed_data

def create_tool_wear_plot(intact_file, fractured_file, output_file, window_size=5):
    """
    Loads tool wear data, applies smoothing, and generates a comparison plot.
    The plot shows both raw and smoothed data for intact and fractured tools
    in two side-by-side subplots and saves it as an SVG file.

    Args:
        intact_file (str): Path to the CSV for the intact tool.
        fractured_file (str): Path to the CSV for the fractured tool.
        output_file (str): The name of the output SVG file.
        window_size (int): The moving average window size for smoothing.
    """
    # --- Style and Font Size ---
    # Set the plot style to match the reference image. This handles grid, background, and borders.
    plt.style.use('seaborn-v0_8-whitegrid')
    # Set a larger global font size for all plot elements as requested.
    plt.rcParams.update({'font.size': 16})

    # --- Load Data ---
    try:
        df_intact = pd.read_csv(intact_file)
        df_fractured = pd.read_csv(fractured_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the CSV files are in the same folder as the script.")
        return

    # --- Data Processing ---
    # Apply the circular moving average smoothing to both datasets.
    df_intact['Smoothed_Pixels'] = circular_moving_average(df_intact['Sum of Pixels'], window_size)
    df_fractured['Smoothed_Pixels'] = circular_moving_average(df_fractured['Sum of Pixels'], window_size)

    # --- Create the Plot ---
    # Create a figure with two subplots (1 row, 2 columns) that share a Y-axis.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # --- Define X-axis Ticks ---
    x_ticks = np.arange(0, 361, 60)

    # --- Configure the Left Plot (Intact Tool) ---
    ax1.plot(df_intact['Degree'], df_intact['Sum of Pixels'], color='green', marker='o', linestyle='none', markersize=4, alpha=0.3)
    ax1.plot(df_intact['Degree'], df_intact['Smoothed_Pixels'], color='green', linestyle='-', linewidth=2.5)
    ax1.set_title('Intact Tool', fontsize=18)
    ax1.set_ylabel('ROI Area [pixels]', fontsize=16)
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(0, 360) # Set precise x-axis limits

    # --- Configure the Right Plot (Fractured Tool) ---
    ax2.plot(df_fractured['Degree'], df_fractured['Sum of Pixels'], color='red', marker='o', linestyle='none', markersize=4, alpha=0.3)
    ax2.plot(df_fractured['Degree'], df_fractured['Smoothed_Pixels'], color='red', linestyle='-', linewidth=2.5)
    ax2.set_title('Fractured Tool', fontsize=18)
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(0, 360) # Set precise x-axis limits

    # --- Global Figure Settings ---
    fig.supxlabel('Angle [°]', fontsize=16)

    # --- Create a Shared Legend ---
    legend_elements = [
        Line2D([0], [0], color='gray', marker='o', linestyle='none', markersize=8, alpha=0.5, label='Raw Data'),
        Line2D([0], [0], color='gray', lw=3, label=f'Smoothed (Window={window_size})')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1, fontsize=14)

    # Adjust layout to prevent labels from overlapping.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save the Figure ---
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Successfully generated plot and saved it as '{output_file}'")


if __name__ == '__main__':
    # --- Configuration ---
    SMOOTHING_WINDOW_SIZE = 5 
    INTACT_DATA_FILE = '../chamfer_intact.csv'
    FRACTURED_DATA_FILE = '../chamfer_2edge_fractured.csv'
    OUTPUT_SVG_FILE = 'tool_wear_comparison_detailed.svg'
    
    # --- Run the script ---
    create_tool_wear_plot(
        intact_file=INTACT_DATA_FILE,
        fractured_file=FRACTURED_DATA_FILE,
        output_file=OUTPUT_SVG_FILE,
        window_size=SMOOTHING_WINDOW_SIZE
    )
