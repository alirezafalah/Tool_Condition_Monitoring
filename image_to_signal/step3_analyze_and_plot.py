import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d # For the wrap-around moving average

def find_roi_and_calculate_area(mask_np, roi_height):
    white_pixel_coords = np.where(mask_np == 255)
    if white_pixel_coords[0].size == 0: return 0
    last_row = white_pixel_coords[0].max()
    first_row = max(0, last_row - roi_height)
    roi = mask_np[first_row:last_row, :]
    return np.sum(roi) / 255

def run(config):
    """
    Analyzes final masks within an ROI, saves data to CSV, filters, and plots.
    """
    input_dir = config['FINAL_MASKS_DIR']
    
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No final masks found in '{input_dir}'. Skipping.")
            return
    except FileNotFoundError:
        print(f"Error: Final masks directory not found at '{input_dir}'.")
        return

    # --- 1. Analyze ROI Area ---
    results = []
    print(f"Starting ROI analysis for {len(image_files)} masks...")
    for filename in tqdm(image_files, desc="Analyzing ROI"):
        try:
            # Extract angle from filename
            angle = float(filename.split('_')[0])
            image_path = os.path.join(input_dir, filename)
            mask_np = np.array(Image.open(image_path))
            roi_area = find_roi_and_calculate_area(mask_np, config['roi_height'])
            results.append({'Angle (Degrees)': angle, 'ROI Area (Pixels)': roi_area})
        except (ValueError, IndexError):
            print(f"Could not parse angle from filename: {filename}. Skipping.")
            continue
            
    if not results:
        print("No data was generated from ROI analysis.")
        return
        
    df = pd.DataFrame(results)
    
    # --- 2. Apply Wrap-Around Moving Average (Optional) ---
    target_column = 'ROI Area (Pixels)' # Default column to use
    if config.get('APPLY_MOVING_AVERAGE', False):
        window_size = config.get('MOVING_AVERAGE_WINDOW', 5)
        print(f"Applying wrap-around moving average with window size {window_size}...")
        
        # Use convolution with 'wrap' mode for an efficient circular moving average
        weights = np.ones(window_size) / window_size
        smoothed_data = convolve1d(df['ROI Area (Pixels)'], weights=weights, mode='wrap')
        
        # Add the smoothed data as a new column and set it as the target for plotting
        df['Smoothed ROI Area'] = smoothed_data
        target_column = 'Smoothed ROI Area'

    # --- 3. Save Data to CSV ---
    csv_path = config['ROI_CSV_PATH']
    csv_dir = os.path.dirname(csv_path)
    if csv_dir: os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"ROI data saved to '{csv_path}'")

    # --- 4. Filter Outliers ---
    mean = df[target_column].mean()
    std = df[target_column].std()
    factor = config['outlier_std_dev_factor']
    inliers = df[(df[target_column] >= mean - factor * std) & (df[target_column] <= mean + factor * std)]
    print(f"Removed {len(df) - len(inliers)} outliers from '{target_column}'.")

    # --- 5. Plot Data ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot raw data as scattered points if smoothing was applied
    if target_column == 'Smoothed ROI Area':
        ax.scatter(inliers['Angle (Degrees)'], inliers['ROI Area (Pixels)'], color='lightgray', s=10, label='Raw Data')

    ax.plot(inliers['Angle (Degrees)'], inliers[target_column], marker='.', linestyle='-', markersize=4, label='Smoothed Data')
    ax.set_title('Tool ROI Area vs. Rotation Angle', fontsize=18, fontweight='bold')
    ax.set_xlabel('Angle (Degrees)', fontsize=14)
    ax.set_ylabel('Projected Area in ROI (Pixel Count)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    plt.tight_layout()
    
    plot_path = config['ROI_PLOT_PATH']
    plot_dir = os.path.dirname(plot_path)
    if plot_dir: os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to '{plot_path}'")
    plt.show()