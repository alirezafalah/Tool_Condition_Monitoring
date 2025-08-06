import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

def circular_moving_average(data, window_size):
    """
    Calculates a circular moving average for a given 1D array of data.
    This is useful for data that is periodic, like angles from 0-360 degrees.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number to ensure symmetry.")
    
    # Pad the data by wrapping the ends to handle the circular nature
    pad_size = (window_size - 1) // 2
    padded_data = np.pad(data, pad_width=pad_size, mode='wrap')
    
    # Use convolution to efficiently calculate the moving average
    weights = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, weights, mode='valid')
    return smoothed_data

def process_and_plot_tool_data(input_file, output_dir, num_segments=4, smoothing_window=15):
    """
    Loads tool wear data, performs scaling, smoothing, shifting, and segmentation,
    and saves a publication-quality plot for each step.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the processed data and SVG plots.
        num_segments (int): The number of segments to divide the data into.
        smoothing_window (int): The window size for the main moving average.
    """
    # --- Setup ---
    # Use a clean and professional style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 16})
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    # =========================================================================
    # Step 0: Initial Plot of Original Data with Light Smoothing
    # =========================================================================
    # Apply a light smoothing (window=5) for initial visualization
    df_lightly_smoothed = df.copy()
    df_lightly_smoothed["Smoothed_Pixels"] = circular_moving_average(df_lightly_smoothed["Sum of Pixels"], window_size=5)

    # --- Plotting Step 0 ---
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.plot(df["Degree"], df["Sum of Pixels"], marker=".", linestyle="none", color='gray', alpha=0.3, label="Raw Data")
    ax0.plot(df_lightly_smoothed["Degree"], df_lightly_smoothed["Smoothed_Pixels"], linestyle="-", color='darkorange', label="Smoothed (Window=5)")
    ax0.set_xlabel("Angle [°]", fontsize=14)
    ax0.set_ylabel("Sum of Pixels", fontsize=14) # Note: Not normalized yet
    # ax0.set_title("Original Signal with Light Smoothing", fontsize=16)
    ax0.set_xlim(0, 360)
    ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)
    fig0.savefig(os.path.join(output_dir, "0_initial_smoothed_signal.svg"), format='svg', bbox_inches='tight')
    print("Saved plot: 0_initial_smoothed_signal.svg")
    plt.close(fig0)


    # =========================================================================
    # Step 1: Scaling and Main Smoothing
    # =========================================================================
    df_scaled = df.copy()
    # Normalize the pixel sum to a 0-1 range for consistent comparison
    df_scaled["Sum of Pixels"] = (df_scaled["Sum of Pixels"] - df_scaled["Sum of Pixels"].min()) / \
                                 (df_scaled["Sum of Pixels"].max() - df_scaled["Sum of Pixels"].min())
    # Apply the main circular moving average to smooth out noise
    df_scaled["Smoothed_Pixels"] = circular_moving_average(df_scaled["Sum of Pixels"], smoothing_window)

    # --- Plotting Step 1 ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_scaled["Degree"], df_scaled["Sum of Pixels"], marker=".", linestyle="none", color='gray', alpha=0.3, label="Raw Data")
    ax1.plot(df_scaled["Degree"], df_scaled["Smoothed_Pixels"], linestyle="-", color='teal', label="Smoothed Data")
    ax1.set_xlabel("Angle [°]", fontsize=14)
    ax1.set_ylabel("Normalized ROI Area", fontsize=14)
    ax1.set_xlim(0, 360)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)
    fig1.savefig(os.path.join(output_dir, "1_normalized_signal.svg"), format='svg', bbox_inches='tight')
    print("Saved plot: 1_normalized_signal.svg")
    plt.close(fig1)

    # =========================================================================
    # Step 2: Shifting the Data to its Minimum
    # =========================================================================
    # Find the minimum point to use as the new 'zero' angle
    min_index = df_scaled["Sum of Pixels"].idxmin()
    # Reorder the dataframe so it starts at the minimum
    df_shifted = pd.concat([df_scaled.loc[min_index:], df_scaled.loc[:min_index - 1]]).reset_index(drop=True)
    # Adjust the degree values to start from 0
    df_shifted["Degree"] = df_shifted["Degree"] - df_shifted.iloc[0]["Degree"]
    df_shifted.loc[df_shifted["Degree"] < 0, "Degree"] += 360
    
    df_shifted.to_csv(os.path.join(output_dir, "processed_tool_data.csv"), index=False)
    print(f"Saved processed data to: {os.path.join(output_dir, 'processed_tool_data.csv')}")

    # --- Plotting Step 2 ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df_shifted["Degree"], df_shifted["Sum of Pixels"], color="gray", marker='.', linestyle='none', alpha=0.3, label="Raw Data")
    ax2.plot(df_shifted["Degree"], df_shifted["Smoothed_Pixels"], color="darkblue", linestyle='-', label="Smoothed Data")
    ax2.set_xlabel("Shifted Angle [°]", fontsize=14)
    ax2.set_ylabel("Normalized ROI Area", fontsize=14)
    ax2.set_xlim(0, 360)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)
    fig2.savefig(os.path.join(output_dir, "2_shifted_signal.svg"), format='svg', bbox_inches='tight')
    print("Saved plot: 2_shifted_signal.svg")
    plt.close(fig2)

    # =========================================================================
    # Step 3: Splitting the Smoothed Data into Segments
    # =========================================================================
    segment_size = len(df_shifted) // num_segments

    # --- Plotting Step 3 ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_segments))
    
    # Plot the original (raw) data in the background for context
    ax3.plot(df_shifted["Degree"], df_shifted["Sum of Pixels"], color="gray", marker='.', linestyle='none', alpha=0.3, label="Raw Data")
    
    for i in range(num_segments):
        segment = df_shifted.iloc[i * segment_size: (i + 1) * segment_size]
        # Plotting the smoothed data for each segment
        ax3.plot(segment["Degree"], segment["Smoothed_Pixels"], color=colors[i], 
                 marker='.', linestyle='-', label=f"Segment {i+1}")

    ax3.set_xlabel("Shifted Angle [°]", fontsize=14)
    ax3.set_ylabel("Normalized ROI Area", fontsize=14)
    # Update legend to include the new "Raw Data" plot
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=num_segments + 1, frameon=False, fontsize=12)
    ax3.set_xlim(0, 360)
    fig3.savefig(os.path.join(output_dir, "3_segmented_signal.svg"), format='svg', bbox_inches='tight')
    print("Saved plot: 3_segmented_signal.svg")
    plt.close(fig3)


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Make sure this path points to your input CSV file.
    INPUT_FILE_PATH = '../chamfer_1edge_fractured.csv'
    
    # Directory where the output plots and data will be saved.
    OUTPUT_DIRECTORY = 'modified_chamfer_1edge_fractured.csv'

    # The number of segments to divide the final data into.
    NUM_SEGMENTS = 2
    
    # The window size for the main moving average. Must be an odd number.
    SMOOTHING_WINDOW = 15
    
    # --- Run the script ---
    process_and_plot_tool_data(
        input_file=INPUT_FILE_PATH,
        output_dir=OUTPUT_DIRECTORY,
        num_segments=NUM_SEGMENTS,
        smoothing_window=SMOOTHING_WINDOW
    )
