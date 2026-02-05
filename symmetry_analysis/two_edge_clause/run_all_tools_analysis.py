"""
Left-Right Symmetry Analysis for 2-Edge Tools
==============================================
This script analyzes tool symmetry by comparing the left and right halves
of each mask image for a specified 90-degree range of rotation.

For 2-edge tools, this captures asymmetry that might be lost when only
analyzing the 1D area-vs-angle profile.

Workflow:
1. Load final masks for all 2-edge tools
2. For each tool, analyze frames from START_FRAME to START_FRAME + 89
3. Find the global ROI (most bottom white pixel across analyzed frames)
4. For each frame, divide the image into left/right halves at tool center
5. Count white pixels in each half (within ROI region)
6. Compare left vs right to detect asymmetry
7. Classify based on threshold and compare to actual condition
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
TOOLS_METADATA_PATH = os.path.join(BASE_DIR, "tools_metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", "left_right_method")

# Starting frame (0 = start from frame 0, 80 = start from frame 80, etc.)
START_FRAME = 0

# Number of frames to analyze (90 degrees)
NUM_FRAMES = 90

# ROI height (rows above the bottom-most white pixel)
ROI_HEIGHT = 200

# Threshold for classification (asymmetry ratio above this = Damaged)
# Set based on tool062 (fractured) with mean ratio 0.033583
ASYMMETRY_THRESHOLD = 0.033

# Output formats: Choose from 'png', 'svg', 'eps', 'jpg' or any combination
OUTPUT_FORMATS = ['png']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tools_metadata():
    """Load the tools metadata CSV into a dictionary keyed by tool_id."""
    metadata = {}
    if os.path.exists(TOOLS_METADATA_PATH):
        with open(TOOLS_METADATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['tool_id']] = row
    return metadata

def get_two_edge_tools(tools_metadata):
    """Get list of tool IDs that have 2 edges AND have mask folders with images."""
    two_edge_tools = []
    for tool_id, meta in tools_metadata.items():
        edges = meta.get('edges', '')
        try:
            if int(edges) == 2:
                # Check if mask folder exists AND has images
                mask_folder = get_mask_folder(tool_id)
                if mask_folder is not None:
                    mask_files = get_mask_files(mask_folder)
                    if len(mask_files) > 0:
                        two_edge_tools.append(tool_id)
        except:
            pass
    return sorted(two_edge_tools)

def get_mask_folder(tool_id):
    """Get the mask folder path for a tool, trying different naming patterns."""
    patterns = [
        f"{tool_id}_final_masks",
        f"{tool_id}gain10paperBG_final_masks",
        f"{tool_id}gain10_final_masks"
    ]
    
    for pattern in patterns:
        folder = os.path.join(MASKS_DIR, pattern)
        if os.path.exists(folder):
            return folder
    
    return None

def get_mask_files(mask_folder):
    """Get all mask files from a folder, sorted by frame number."""
    # Get all tiff files
    pattern = os.path.join(mask_folder, "*.tiff")
    files = glob.glob(pattern)
    
    if not files:
        pattern = os.path.join(mask_folder, "*.tif")
        files = glob.glob(pattern)
    
    if not files:
        return []
    
    # Sort by frame number (extract number from filename)
    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
        parts = basename.replace('.tiff', '').replace('.tif', '').split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    
    files = sorted(files, key=extract_frame_num)
    return files

def find_global_roi_bottom(mask_files, start_frame, num_frames):
    """Find the most bottom white pixel across analyzed frames (global ROI)."""
    global_bottom = 0
    
    end_frame = min(start_frame + num_frames, len(mask_files))
    
    for i in range(start_frame, end_frame):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) > 0:
            bottom_row = np.max(white_pixels[0])
            global_bottom = max(global_bottom, bottom_row)
    
    return global_bottom

def analyze_left_right_symmetry(mask_path, global_roi_bottom, roi_height):
    """
    Analyze a single mask image for left-right symmetry.
    
    The center column is determined by the center of the tool's white pixel contour,
    not the image center. The ROI is defined as roi_height rows above global_roi_bottom.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    height, width = mask.shape
    
    # Define ROI region
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    
    # Extract ROI region
    roi_mask = mask[roi_top:roi_bottom, :]
    
    # Find the center column of the tool's white pixels
    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return None
    
    left_col = np.min(white_pixels[1])
    right_col = np.max(white_pixels[1])
    center_col = (left_col + right_col) // 2
    
    # Divide into left and right halves based on tool center
    left_half = roi_mask[:, left_col:center_col]
    right_half = roi_mask[:, center_col + 1:right_col + 1]
    
    # Count white pixels in each half
    left_count = np.sum(left_half == 255)
    right_count = np.sum(right_half == 255)
    
    # Calculate difference and ratio
    diff = abs(left_count - right_count)
    total = left_count + right_count
    ratio = diff / total if total > 0 else 0
    
    # Normalized difference
    half_width = center_col - left_col
    half_area = roi_height * half_width if half_width > 0 else 1
    normalized_diff = diff / half_area
    
    return {
        'left_count': left_count,
        'right_count': right_count,
        'difference': diff,
        'ratio': ratio,
        'normalized_diff': normalized_diff,
        'center_col': center_col
    }

def analyze_tool(tool_id, mask_files, start_frame, num_frames, roi_height):
    """Analyze a single tool and return statistics and frame-by-frame data."""
    end_frame = min(start_frame + num_frames, len(mask_files))
    actual_frames = end_frame - start_frame
    
    if actual_frames < 10:
        return None
    
    # Find global ROI bottom for this tool's analyzed frames
    global_roi_bottom = find_global_roi_bottom(mask_files, start_frame, num_frames)
    
    if global_roi_bottom == 0:
        return None
    
    # Analyze each frame
    frame_data = []
    
    for i in range(start_frame, end_frame):
        result = analyze_left_right_symmetry(mask_files[i], global_roi_bottom, roi_height)
        if result:
            frame_data.append({
                'frame': i,
                'left_count': result['left_count'],
                'right_count': result['right_count'],
                'difference': result['difference'],
                'ratio': result['ratio'],
                'center_col': result['center_col']
            })
    
    if not frame_data:
        return None
    
    ratios = [f['ratio'] for f in frame_data]
    diffs = [f['difference'] for f in frame_data]
    
    return {
        'mean_ratio': np.mean(ratios),
        'max_ratio': np.max(ratios),
        'std_ratio': np.std(ratios),
        'mean_diff': np.mean(diffs),
        'frames_analyzed': len(frame_data),
        'global_roi_bottom': global_roi_bottom,
        'frame_data': frame_data  # For plotting
    }

def plot_tool_analysis(tool_id, stats, condition, output_dir):
    """Plot individual tool left-right analysis results."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    frame_data = stats['frame_data']
    frames = [f['frame'] for f in frame_data]
    left_counts = [f['left_count'] for f in frame_data]
    right_counts = [f['right_count'] for f in frame_data]
    differences = [f['difference'] for f in frame_data]
    ratios = [f['ratio'] for f in frame_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Left vs Right pixel counts
    ax1 = axes[0, 0]
    ax1.plot(frames, left_counts, label='Left Half', color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(frames, right_counts, label='Right Half', color='red', alpha=0.7, linewidth=1.5)
    ax1.set_title(f'{tool_id}: White Pixel Count per Half')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('White Pixel Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute difference
    ax2 = axes[0, 1]
    ax2.plot(frames, differences, color='purple', linewidth=1.5)
    ax2.fill_between(frames, differences, color='purple', alpha=0.2)
    ax2.set_title('Absolute Difference (Left - Right)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Pixel Count Difference')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratio over frames
    ax3 = axes[1, 0]
    ax3.plot(frames, ratios, color='green', linewidth=1.5)
    ax3.axhline(y=ASYMMETRY_THRESHOLD, color='red', linestyle='--', linewidth=1.5, 
                label=f'Threshold ({ASYMMETRY_THRESHOLD})')
    ax3.set_title('Asymmetry Ratio per Frame')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Ratio (|L-R| / Total)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Tool ID: {tool_id}
    Condition: {condition}
    
    Frame Range: {frames[0]} - {frames[-1]}
    Frames Analyzed: {stats['frames_analyzed']}
    
    Mean Asymmetry Ratio: {stats['mean_ratio']:.4f}
    Max Asymmetry Ratio: {stats['max_ratio']:.4f}
    Std Asymmetry Ratio: {stats['std_ratio']:.4f}
    Mean Pixel Difference: {stats['mean_diff']:.1f}
    
    Threshold: {ASYMMETRY_THRESHOLD}
    Prediction: {"Damaged" if stats['mean_ratio'] > ASYMMETRY_THRESHOLD else "Good"}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.set_title('Summary Statistics')
    
    fig.suptitle(f'{tool_id} Left-Right Symmetry Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_left_right_analysis.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

def plot_summary(results_df, output_dir, threshold):
    """Plot the summary bar chart for all tools."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color by actual condition
    colors = []
    for _, row in results_df.iterrows():
        cond = str(row['Condition']).lower()
        if cond in ['fractured', 'deposit']:
            colors.append('red')
        elif cond == 'used':
            colors.append('orange')
        elif cond == 'new':
            colors.append('green')
        else:
            colors.append('gray')
    
    bars = ax.bar(results_df['Tool ID'], results_df['Mean Ratio'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    ax.set_title(f'Left-Right Symmetry Analysis: Mean Asymmetry Ratio by Tool\n(Frames {START_FRAME}-{START_FRAME + NUM_FRAMES - 1})')
    ax.set_ylabel('Mean Asymmetry Ratio')
    ax.set_xlabel('Tool ID')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend for conditions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='New'),
        Patch(facecolor='orange', alpha=0.7, label='Used'),
        Patch(facecolor='red', alpha=0.7, label='Fractured/Deposit'),
        Patch(facecolor='gray', alpha=0.7, label='Unknown'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'left_right_summary.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

def plot_sample_frames(mask_files, global_roi_bottom, roi_height, tool_id, output_dir, start_frame, num_frames):
    """Plot sample frames showing the left-right division within ROI."""
    # Select 5 sample frames spread across the analyzed range
    end_frame = min(start_frame + num_frames, len(mask_files))
    actual_frames = end_frame - start_frame
    sample_indices = [start_frame + int(i * actual_frames / 5) for i in range(5)]
    
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 6))
    
    if n_samples == 1:
        axes = [axes]
    
    roi_top = max(0, global_roi_bottom - roi_height)
    
    for idx, frame_idx in enumerate(sample_indices):
        if frame_idx >= len(mask_files):
            continue
        
        mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        height, width = mask.shape
        
        # Find tool center from white pixels in ROI
        roi_mask = mask[roi_top:global_roi_bottom + 1, :]
        white_pixels = np.where(roi_mask == 255)
        
        if len(white_pixels[1]) > 0:
            left_col = np.min(white_pixels[1])
            right_col = np.max(white_pixels[1])
            center_col = (left_col + right_col) // 2
        else:
            center_col = width // 2
            left_col = 0
            right_col = width - 1
        
        # Create visualization with ROI region and center line
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Draw ROI region (top and bottom horizontal lines)
        cv2.line(vis, (0, roi_top), (width, roi_top), (0, 255, 0), 2)
        cv2.line(vis, (0, global_roi_bottom), (width, global_roi_bottom), (0, 255, 0), 2)
        
        # Draw tool boundary lines (vertical, at left and right edges of tool)
        cv2.line(vis, (left_col, roi_top), (left_col, global_roi_bottom), (255, 255, 0), 1)
        cv2.line(vis, (right_col, roi_top), (right_col, global_roi_bottom), (255, 255, 0), 1)
        
        # Draw center line (vertical, red, at tool center)
        cv2.line(vis, (center_col, roi_top), (center_col, global_roi_bottom), (255, 0, 0), 2)
        
        axes[idx].imshow(vis)
        axes[idx].set_title(f'Frame {frame_idx}°\nCenter: col {center_col}')
        axes[idx].axis('off')
    
    fig.suptitle(f'{tool_id}: Sample Frames\nROI (green), Tool bounds (yellow), Center (red)', fontsize=12)
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_sample_frames.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

def calculate_accuracy(results_df):
    """Calculate accuracy metrics comparing prediction to actual condition."""
    # Define ground truth: fractured/deposit = Damaged, new/used = Good
    def get_ground_truth(condition):
        cond = str(condition).lower()
        if cond in ['fractured', 'deposit']:
            return 'Damaged'
        elif cond in ['new', 'used']:
            return 'Good'
        else:
            return 'Unknown'
    
    results_df['Ground Truth'] = results_df['Condition'].apply(get_ground_truth)
    
    # Filter out unknown
    known = results_df[results_df['Ground Truth'] != 'Unknown']
    
    if len(known) == 0:
        return 0, 0, 0, 0
    
    # Calculate metrics
    correct = (known['Prediction'] == known['Ground Truth']).sum()
    total = len(known)
    accuracy = correct / total
    
    # True positives, false positives, etc.
    tp = ((known['Prediction'] == 'Damaged') & (known['Ground Truth'] == 'Damaged')).sum()
    fp = ((known['Prediction'] == 'Damaged') & (known['Ground Truth'] == 'Good')).sum()
    tn = ((known['Prediction'] == 'Good') & (known['Ground Truth'] == 'Good')).sum()
    fn = ((known['Prediction'] == 'Good') & (known['Ground Truth'] == 'Damaged')).sum()
    
    return accuracy, tp, fp, tn, fn

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("LEFT-RIGHT SYMMETRY ANALYSIS FOR ALL 2-EDGE TOOLS")
    print("=" * 70)
    print(f"Frame Range: {START_FRAME} to {START_FRAME + NUM_FRAMES - 1}")
    print(f"ROI Height: {ROI_HEIGHT} rows")
    print(f"Asymmetry Threshold: {ASYMMETRY_THRESHOLD}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    tools_meta = load_tools_metadata()
    two_edge_tools = get_two_edge_tools(tools_meta)
    
    print(f"\nFound {len(two_edge_tools)} 2-edge tools in metadata.")
    
    results = []
    
    for tool_id in two_edge_tools:
        meta = tools_meta.get(tool_id, {})
        condition = meta.get('condition', 'N/A')
        tool_type = meta.get('type', 'N/A')
        
        # Get mask folder
        mask_folder = get_mask_folder(tool_id)
        if not mask_folder:
            print(f"  {tool_id}: No mask folder found, skipping.")
            continue
        
        # Get mask files
        mask_files = get_mask_files(mask_folder)
        if len(mask_files) < START_FRAME + 10:
            print(f"  {tool_id}: Not enough frames ({len(mask_files)} total), skipping.")
            continue
        
        print(f"Processing {tool_id} (Type: {tool_type}, Condition: {condition})...", end=" ")
        
        # Analyze
        stats = analyze_tool(tool_id, mask_files, START_FRAME, NUM_FRAMES, ROI_HEIGHT)
        
        if stats is None:
            print("FAILED")
            continue
        
        # Classification
        prediction = "Damaged" if stats['mean_ratio'] > ASYMMETRY_THRESHOLD else "Good"
        
        print(f"Mean Ratio: {stats['mean_ratio']:.4f} -> {prediction}")
        
        # Generate individual tool plot
        plot_tool_analysis(tool_id, stats, condition, OUTPUT_DIR)
        
        # Generate sample frames visualization
        plot_sample_frames(mask_files, stats['global_roi_bottom'], ROI_HEIGHT, 
                          tool_id, OUTPUT_DIR, START_FRAME, NUM_FRAMES)
        
        results.append({
            'Tool ID': tool_id,
            'Type': tool_type,
            'Condition': condition,
            'Mean Ratio': round(stats['mean_ratio'], 6),
            'Max Ratio': round(stats['max_ratio'], 6),
            'Std Ratio': round(stats['std_ratio'], 6),
            'Mean Diff': round(stats['mean_diff'], 2),
            'Frames Analyzed': stats['frames_analyzed'],
            'Prediction': prediction
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("\nNo tools were successfully analyzed!")
        return
    
    # Sort by Mean Ratio
    results_df_sorted = results_df.sort_values('Mean Ratio', ascending=True)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'left_right_analysis_results.csv')
    results_df_sorted.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate summary plot
    plot_summary(results_df_sorted, OUTPUT_DIR, ASYMMETRY_THRESHOLD)
    print("Summary plot saved.")
    
    # Calculate accuracy
    accuracy, tp, fp, tn, fn = calculate_accuracy(results_df_sorted.copy())
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df_sorted.to_string(index=False))
    print("=" * 70)
    
    print(f"\n--- Classification Performance ---")
    print(f"Threshold: {ASYMMETRY_THRESHOLD}")
    print(f"Accuracy: {accuracy:.2%} ({tp + tn}/{tp + tn + fp + fn})")
    print(f"True Positives (Damaged correctly detected): {tp}")
    print(f"False Positives (Good marked as Damaged): {fp}")
    print(f"True Negatives (Good correctly detected): {tn}")
    print(f"False Negatives (Damaged missed): {fn}")
    
    # Save metadata
    analysis_meta = {
        'method': 'Left-Right Symmetry Analysis',
        'description': 'Compares left and right halves of tool within ROI for 90-degree range',
        'start_frame': START_FRAME,
        'num_frames': NUM_FRAMES,
        'roi_height': ROI_HEIGHT,
        'threshold': ASYMMETRY_THRESHOLD,
        'accuracy': float(accuracy),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_tools_analyzed': len(results)
    }
    
    meta_path = os.path.join(OUTPUT_DIR, 'analysis_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(analysis_meta, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
