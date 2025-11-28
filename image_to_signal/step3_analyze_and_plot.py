import os
import json
from datetime import datetime
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
            
            # Count white pixels to detect bad segmentation
            total_white_pixels = np.sum(mask_np == 255)
            total_pixels = mask_np.size
            white_ratio = total_white_pixels / total_pixels
            
            roi_area = find_roi_and_calculate_area(mask_np, config['roi_height'])
            results.append({
                'Angle (Degrees)': angle, 
                'ROI Area (Pixels)': roi_area,
                'white_ratio': white_ratio
            })
        except (ValueError, IndexError):
            print(f"Could not parse angle from filename: {filename}. Skipping.")
            continue
            
    if not results:
        print("No data was generated from ROI analysis.")
        return
        
    df = pd.DataFrame(results)
    
    # --- 1.5. Detect and fix outliers (bad segmentation) ---
    outlier_threshold = config.get('WHITE_RATIO_OUTLIER_THRESHOLD', 0.8)
    df['is_outlier'] = df['white_ratio'] > outlier_threshold
    num_outliers = df['is_outlier'].sum()
    
    if num_outliers > 0:
        print(f"\n⚠️  Detected {num_outliers} over-segmented masks (>{outlier_threshold*100:.0f}% white)")
        print(f"   Interpolating with neighboring values...")
        
        for idx in df[df['is_outlier']].index:
            prev_idx = (idx - 1) % len(df)
            next_idx = (idx + 1) % len(df)
            
            if not df.loc[prev_idx, 'is_outlier'] and not df.loc[next_idx, 'is_outlier']:
                avg_value = (df.loc[prev_idx, 'ROI Area (Pixels)'] + df.loc[next_idx, 'ROI Area (Pixels)']) / 2
                df.loc[idx, 'ROI Area (Pixels)'] = avg_value
    else:
        print(f"✓ No over-segmented masks detected")
    
    # Drop helper columns
    df = df[['Angle (Degrees)', 'ROI Area (Pixels)']]
    
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
    
    # --- 3.5. Save Metadata ---
    metadata_dir = os.path.join(os.path.dirname(csv_path), 'analysis_metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    tool_id = os.path.basename(csv_path).replace('_area_vs_angle.csv', '')
    metadata = {
        "tool_id": tool_id,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_analyzed": len(image_files),
        "analysis_type": "raw",
        
        "roi_parameters": {
            "roi_height": config['roi_height']
        },
        
        "processing_parameters": {
            "apply_moving_average": config.get('APPLY_MOVING_AVERAGE', False),
            "moving_average_window": config.get('MOVING_AVERAGE_WINDOW', 5)
        },
        
        "image_processing": {
            "blur_kernel": config['blur_kernel'],
            "closing_kernel": config['closing_kernel'],
            "background_subtraction_method": config['BACKGROUND_SUBTRACTION_METHOD']
        },
        
        "paths": {
            "raw_dir": config['RAW_DIR'],
            "blurred_dir": config['BLURRED_DIR'],
            "masks_dir": config['FINAL_MASKS_DIR']
        }
    }
    
    metadata_path = os.path.join(metadata_dir, f'{tool_id}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to '{metadata_path}'")

    # --- 4. Plot Data ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f'ROI Analysis - {tool_id}')
    
    # Plot raw data as scattered points if smoothing was applied
    if target_column == 'Smoothed ROI Area':
        ax.scatter(df['Angle (Degrees)'], df['ROI Area (Pixels)'], color='lightgray', s=10, label='Raw Data')

    ax.plot(df['Angle (Degrees)'], df[target_column], marker='.', linestyle='-', markersize=4, label='Smoothed Data')
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
    
    plt.savefig(plot_path, format='svg', dpi=300)
    print(f"Plot saved in SVG format to '{plot_path}'")
    plt.close()
    
    # Open the saved plot with default system application (e.g., Edge)
    import subprocess
    try:
        subprocess.Popen(['cmd', '/c', 'start', '', plot_path], shell=False)
        print(f"Opening plot with default application...")
    except Exception as e:
        print(f"Could not open plot automatically: {e}")