import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def find_roi_and_calculate_area(mask_np, roi_height):
    """Calculate ROI area for a mask."""
    white_pixel_coords = np.where(mask_np == 255)
    if white_pixel_coords[0].size == 0: return 0
    last_row = white_pixel_coords[0].max()
    first_row = max(0, last_row - roi_height)
    roi = mask_np[first_row:last_row, :]
    return np.sum(roi) / 255

def run(config):
    """
    Analyzes final masks, applies preprocessing (scale 0-1, shift minimum to 0°),
    segments by number of peaks, and saves processed data.
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
    print(f"Starting ROI analysis for {len(image_files)} masks (processed mode)...")
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
    
    # --- 2. Scale to 0-1 ---
    print("Scaling data to 0-1 range...")
    min_val = df['ROI Area (Pixels)'].min()
    max_val = df['ROI Area (Pixels)'].max()
    df['ROI Area (Pixels)'] = (df['ROI Area (Pixels)'] - min_val) / (max_val - min_val)
    
    # --- 3. Shift Minimum to 0° ---
    print("Shifting minimum value to 0 degrees...")
    min_index = df['ROI Area (Pixels)'].idxmin()
    df_shifted = pd.concat([df.loc[min_index:], df.loc[:min_index - 1]])
    df_shifted = df_shifted.reset_index(drop=True)
    
    # Shift the degree axis to start from 0
    df_shifted['Angle (Degrees)'] = df_shifted['Angle (Degrees)'] - df_shifted.iloc[0]['Angle (Degrees)']
    df_shifted.loc[df_shifted['Angle (Degrees)'] < 0, 'Angle (Degrees)'] += 360
    
    # --- 4. Save Processed Data to CSV ---
    csv_path = config['PROCESSED_CSV_PATH']
    csv_dir = os.path.dirname(csv_path)
    if csv_dir: os.makedirs(csv_dir, exist_ok=True)
    df_shifted.to_csv(csv_path, index=False)
    print(f"Processed data saved to '{csv_path}'")
    
    # --- 4.5. Save Metadata ---
    metadata_dir = os.path.join(os.path.dirname(csv_path), 'analysis_metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    tool_id = os.path.basename(csv_path).replace('_area_vs_angle_processed.csv', '')
    metadata = {
        "tool_id": tool_id,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_analyzed": len(image_files),
        "analysis_type": "processed",
        
        "roi_parameters": {
            "roi_height": config['roi_height'],
            "images_for_366_deg": config['images_for_366_deg']
        },
        
        "processing_parameters": {
            "number_of_peaks": config.get('NUMBER_OF_PEAKS', 1),
            "outlier_std_dev_factor": config['outlier_std_dev_factor']
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
    
    metadata_path = os.path.join(metadata_dir, f'{tool_id}_processed_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Processed metadata saved to '{metadata_path}'")
    
    # --- 5. Create Segmented Plot ---
    num_segments = config.get('NUMBER_OF_PEAKS', 1)
    segment_size = len(df_shifted) // num_segments
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(df_shifted)
        segment = df_shifted.iloc[start_idx:end_idx]
        ax.scatter(segment['Angle (Degrees)'], segment['ROI Area (Pixels)'], 
                  color=colors[i % len(colors)], s=20, label=f'Segment {i+1}', alpha=0.7)
    
    ax.set_title(f'Processed Tool Profile - {num_segments} Segments', fontsize=18, fontweight='bold')
    ax.set_xlabel('Angle (Degrees)', fontsize=14)
    ax.set_ylabel('Normalized ROI Area (0-1)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    plt.tight_layout()
    
    plot_path = config['PROCESSED_PLOT_PATH']
    plot_dir = os.path.dirname(plot_path)
    if plot_dir: os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(plot_path, format='svg', dpi=300)
    print(f"Processed plot saved to '{plot_path}'")
    plt.close()  # Close instead of show to avoid blocking
