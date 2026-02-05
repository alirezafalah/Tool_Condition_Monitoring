import os
import json
import csv
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils.optimized_processing import analyze_roi, print_optimization_header

def run(config):
    """
    Analyzes final masks, applies preprocessing (scale 0-1, shift minimum to 0°),
    segments by number of peaks, and saves processed data.
    Uses pipeline-wide optimization method.
    """
    input_dir = config['FINAL_MASKS_DIR']
    optimization_method = config.get('OPTIMIZATION_METHOD', 'gpu')
    
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No final masks found in '{input_dir}'. Skipping.")
            return
    except FileNotFoundError:
        print(f"Error: Final masks directory not found at '{input_dir}'.")
        return

    # Display optimization info
    print_optimization_header(optimization_method, f"Step 4: Processed Analysis ({len(image_files)} masks)")

    # --- 1. Analyze ROI Area with optimized processing ---
    start_time = time.time()
    
    results, failed = analyze_roi(
        image_files=image_files,
        input_dir=input_dir,
        roi_height=config['roi_height'],
        method=optimization_method
    )
    
    duration = time.time() - start_time
    print(f"\n✅ ROI analysis complete: {len(results)}/{len(image_files)} masks in {duration:.2f}s")
    
    if failed:
        print(f"⚠️ Failed: {len(failed)} files")
        for error in failed[:3]:
            print(f"   - {error}")
            
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
    
    # Load tool metadata from tools_metadata.csv
    tool_meta = None
    tools_metadata_path = os.path.join(os.path.dirname(csv_path), '..', 'tools_metadata.csv')
    if os.path.exists(tools_metadata_path):
        with open(tools_metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['tool_id'] == tool_id:
                    tool_meta = row
                    break
    
    metadata = {
        "tool_id": tool_id,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_analyzed": len(image_files),
        "analysis_type": "processed",
        
        "tool_metadata": tool_meta if tool_meta else {},
        
        "roi_parameters": {
            "roi_height": config['roi_height']
        },
        
        "processing_parameters": {
            "number_of_peaks": config.get('NUMBER_OF_PEAKS', 1),
            "white_ratio_outlier_threshold": config.get('WHITE_RATIO_OUTLIER_THRESHOLD', 0.8)
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
    fig.canvas.manager.set_window_title(f'Processed Profile - {tool_id}')
    
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
    plt.close()
    
    # Open the saved plot with default system application (e.g., Edge)
    import subprocess
    try:
        subprocess.Popen(['cmd', '/c', 'start', '', plot_path], shell=False)
        print(f"Opening plot with default application...")
    except Exception as e:
        print(f"Could not open plot automatically: {e}")
