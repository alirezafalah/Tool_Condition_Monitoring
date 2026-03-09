"""
Update Single Tool Analysis
============================
Re-runs left-right symmetry analysis for a single tool with a given ROI,
then updates the existing CSV, summary plot, and metadata JSON in-place.
Only recalculates the specified tool — everything else stays untouched.

Usage:
    python update_single_tool.py <tool_id> <roi_height>
    
Examples:
    python update_single_tool.py tool072 300
    python update_single_tool.py tool002 500
"""

import sys
import os

# Reuse everything from the main analysis script
from run_all_tools_analysis import (
    BASE_DIR, MASKS_DIR, TOOLS_METADATA_PATH, OUTPUT_DIR,
    START_FRAME, NUM_FRAMES, ASYMMETRY_THRESHOLD,
    WHITE_RATIO_OUTLIER_THRESHOLD, OUTPUT_FORMATS,
    ROTATION_ANGLE_DEG,
    load_tools_metadata, get_mask_folder, get_mask_files,
    get_largest_contour_mask, find_global_roi_bottom,
    analyze_left_right_symmetry, plot_tool_analysis,
    plot_sample_frames, plot_summary, calculate_accuracy,
)

import numpy as np
import pandas as pd
import json
from datetime import datetime


def analyze_tool_with_roi(tool_id, roi_height):
    """Analyze a single tool with a specific ROI height. Returns stats dict or None."""
    mask_folder = get_mask_folder(tool_id)
    if not mask_folder:
        return None, "No mask folder found"

    mask_files = get_mask_files(mask_folder)
    if len(mask_files) < START_FRAME + 10:
        return None, f"Not enough frames ({len(mask_files)})"

    global_roi_bottom = find_global_roi_bottom(mask_files, START_FRAME, NUM_FRAMES, roi_height)
    if global_roi_bottom == 0:
        return None, "Could not find ROI bottom"

    end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))

    frame_data = []
    for i in range(START_FRAME, end_frame):
        result = analyze_left_right_symmetry(mask_files[i], global_roi_bottom, roi_height)
        if result:
            frame_data.append({
                'frame': i,
                'left_count': result['left_count'],
                'right_count': result['right_count'],
                'difference': result['difference'],
                'ratio': result['ratio'],
                'center_col': result['center_col'],
            })

    if not frame_data:
        return None, "No valid frames"

    ratios = [f['ratio'] for f in frame_data]
    diffs = [f['difference'] for f in frame_data]

    stats = {
        'mean_ratio': np.mean(ratios),
        'max_ratio': np.max(ratios),
        'std_ratio': np.std(ratios),
        'mean_diff': np.mean(diffs),
        'frames_analyzed': len(frame_data),
        'global_roi_bottom': global_roi_bottom,
        'roi_height': roi_height,
        'frame_data': frame_data,
    }
    return stats, None


def main():
    # ---- parse arguments ----
    if len(sys.argv) != 3:
        print("Usage: python update_single_tool.py <tool_id> <roi_height>")
        print("Example: python update_single_tool.py tool072 300")
        sys.exit(1)

    tool_id = sys.argv[1]
    try:
        roi_height = int(sys.argv[2])
    except ValueError:
        print(f"Error: roi_height must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    csv_path = os.path.join(OUTPUT_DIR, 'left_right_analysis_results.csv')
    meta_path = os.path.join(OUTPUT_DIR, 'analysis_metadata.json')

    if not os.path.exists(csv_path):
        print(f"Error: results CSV not found at {csv_path}")
        print("Run run_all_tools_analysis.py first to create the initial results.")
        sys.exit(1)

    # ---- load metadata ----
    tools_meta = load_tools_metadata()
    meta = tools_meta.get(tool_id, {})
    condition = meta.get('condition', 'N/A')
    tool_type = meta.get('type', 'N/A')

    print("=" * 60)
    print(f"UPDATING: {tool_id}  (ROI={roi_height})")
    print(f"Type: {tool_type}, Condition: {condition}")
    print("=" * 60)

    # ---- run analysis ----
    stats, error = analyze_tool_with_roi(tool_id, roi_height)
    if error:
        print(f"FAILED: {error}")
        sys.exit(1)

    prediction = "Damaged" if stats['mean_ratio'] > ASYMMETRY_THRESHOLD else "Good"
    print(f"Mean Ratio: {stats['mean_ratio']:.6f}  ->  {prediction}")
    print(f"Frames Analyzed: {stats['frames_analyzed']}")

    # ---- build the new row ----
    new_row = {
        'Tool ID': tool_id,
        'Type': tool_type,
        'Condition': condition,
        'ROI Height': roi_height,
        'Mean Ratio': round(stats['mean_ratio'], 6),
        'Max Ratio': round(stats['max_ratio'], 6),
        'Std Ratio': round(stats['std_ratio'], 6),
        'Mean Diff': round(stats['mean_diff'], 2),
        'Frames Analyzed': stats['frames_analyzed'],
        'Prediction': prediction,
    }

    # ---- update CSV ----
    results_df = pd.read_csv(csv_path)

    if tool_id in results_df['Tool ID'].values:
        # Replace existing row
        idx = results_df.index[results_df['Tool ID'] == tool_id][0]
        for col, val in new_row.items():
            results_df.at[idx, col] = val
        print(f"\nUpdated existing row for {tool_id} in CSV.")
    else:
        # Append new row
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"\nAdded new row for {tool_id} to CSV.")

    # Re-sort by Mean Ratio
    results_df = results_df.sort_values('Mean Ratio', ascending=True).reset_index(drop=True)
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # ---- regenerate individual tool plots ----
    mask_folder = get_mask_folder(tool_id)
    mask_files = get_mask_files(mask_folder)

    plot_tool_analysis(tool_id, stats, condition, OUTPUT_DIR)
    print(f"Updated: {tool_id}_left_right_analysis.png")

    plot_sample_frames(mask_files, stats['global_roi_bottom'], stats['roi_height'],
                       tool_id, OUTPUT_DIR, START_FRAME, NUM_FRAMES)
    print(f"Updated: {tool_id}_sample_frames.png")

    # ---- regenerate summary plot ----
    plot_summary(results_df, OUTPUT_DIR, ASYMMETRY_THRESHOLD)
    print(f"Updated: left_right_summary.png")

    # ---- recalculate accuracy & update JSON ----
    accuracy, tp, fp, tn, fn = calculate_accuracy(results_df.copy())

    analysis_meta = {
        'method': 'Left-Right Symmetry Analysis',
        'description': 'Compares left and right halves of tool within ROI for 90-degree range',
        'start_frame': START_FRAME,
        'num_frames': NUM_FRAMES,
        'threshold': ASYMMETRY_THRESHOLD,
        'accuracy': float(accuracy),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_tools_analyzed': len(results_df),
        'last_updated_tool': tool_id,
        'last_updated_roi': roi_height,
    }

    with open(meta_path, 'w') as f:
        json.dump(analysis_meta, f, indent=2)
    print(f"Updated: analysis_metadata.json")

    # ---- final summary ----
    print("\n" + "=" * 60)
    print("UPDATED RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.2%} ({tp + tn}/{tp + tn + fp + fn})")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 60)


if __name__ == "__main__":
    main()
