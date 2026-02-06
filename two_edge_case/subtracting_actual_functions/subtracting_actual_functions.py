"""
Symmetry Analysis Method for Tool Condition Monitoring
=======================================================
This script analyzes tool profiles by comparing the two 180° segments of 2-edge tools.
Asymmetry indicates potential damage.

Output: DATA/threshold_analysis/symmetry_method/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import glob
import csv
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
PROFILES_DIR = os.path.join(BASE_DIR, "1d_profiles")
TOOLS_METADATA_PATH = os.path.join(BASE_DIR, "tools_metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", "subtracting_actual_functions")

# Threshold for classification (can be tuned based on probability analysis)
SYMMETRY_THRESHOLD = 0.035

# Output formats: Choose from 'png', 'svg', 'eps', 'jpg' or any combination
# Examples: ['svg'], ['png', 'eps'], ['svg', 'eps', 'png', 'jpg']
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

def get_available_tools():
    """Find all tools with processed data in 1d_profiles."""
    pattern = os.path.join(PROFILES_DIR, "tool*_processed_data.csv")
    files = glob.glob(pattern)
    tools = []
    for f in files:
        basename = os.path.basename(f)
        tool_id = basename.replace("_processed_data.csv", "")
        tools.append({
            'tool_id': tool_id,
            'filepath': f
        })
    return sorted(tools, key=lambda x: x['tool_id'])

def load_and_interpolate(filepath, grid_step=1.0):
    """Loads the CSV data and interpolates it to a regular grid from 0 to 360 degrees."""
    df = pd.read_csv(filepath)
    angle_col = 'Angle (Degrees)'
    area_col = 'ROI Area (Pixels)'
    
    if angle_col not in df.columns or area_col not in df.columns:
        raise ValueError(f"Required columns not found in {filepath}")
    
    angles = df[angle_col].values
    areas = df[area_col].values
    
    sorted_indices = np.argsort(angles)
    angles = angles[sorted_indices]
    areas = areas[sorted_indices]
    
    f = interp1d(angles, areas, kind='linear', fill_value="extrapolate")
    grid_angles = np.arange(0, 360, grid_step)
    interpolated_areas = f(grid_angles)
    
    return grid_angles, interpolated_areas

def analyze_symmetry(angles, values):
    """Splits the data into two 180-degree segments and compares them."""
    n_points = len(values)
    mid_point = n_points // 2
    
    part1 = values[:mid_point]
    part2 = values[mid_point:]
    
    min_len = min(len(part1), len(part2))
    part1 = part1[:min_len]
    part2 = part2[:min_len]
    
    diff = np.abs(part1 - part2)
    
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(diff)
    
    return part1, part2, diff, mae, rmse, max_diff

def plot_symmetry(tool_id, grid_angles, part1, part2, diff, output_dir):
    """Plots the comparison and difference."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    n_points = len(part1)
    plot_angles = grid_angles[:n_points]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(plot_angles, part1, label='0-180° Segment', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(plot_angles, part2, label='180-360° Segment', color='red', alpha=0.7, linestyle='--', linewidth=2)
    ax1.set_title(f'{tool_id} Symmetry Analysis\nOverlay of 180° Segments')
    ax1.set_ylabel('ROI Area (Pixels)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(plot_angles, diff, label='Absolute Difference', color='purple', linewidth=2)
    ax2.set_title('Difference (Absolute Error)')
    ax2.set_xlabel('Angle (Degrees within segment)')
    ax2.set_ylabel('Difference')
    ax2.fill_between(plot_angles, diff, color='purple', alpha=0.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_symmetry_plot.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

def plot_summary(results_df, output_dir, threshold):
    """Plots the summary bar chart for all tools."""
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
    
    bars = ax.bar(results_df['Tool ID'], results_df['MAE'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    ax.set_title('Symmetry Analysis: Mean Absolute Error by Tool')
    ax.set_ylabel('Mean Absolute Error (MAE)')
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
        path = os.path.join(output_dir, f'symmetry_summary.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SYMMETRY ANALYSIS FOR TOOL CONDITION MONITORING")
    print("=" * 70)
    print(f"Profiles Directory: {PROFILES_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Threshold: {SYMMETRY_THRESHOLD}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    tools_meta = load_tools_metadata()
    
    # Get available tools
    tools = get_available_tools()
    print(f"Found {len(tools)} tools with processed data.\n")
    
    results = []
    
    for tool in tools:
        tool_id = tool['tool_id']
        filepath = tool['filepath']
        
        # Get metadata
        meta = tools_meta.get(tool_id, {})
        edges = meta.get('edges', 'N/A')
        condition = meta.get('condition', 'N/A')
        tool_type = meta.get('type', 'N/A')
        
        print(f"Processing {tool_id} (Type: {tool_type}, Edges: {edges}, Condition: {condition})...")
        
        try:
            grid_angles, values = load_and_interpolate(filepath)
            part1, part2, diff, mae, rmse, max_diff = analyze_symmetry(grid_angles, values)
            
            # Classification
            prediction = "Damaged" if mae > SYMMETRY_THRESHOLD else "Good"
            
            print(f"  MAE: {mae:.4f} -> Prediction: {prediction}")
            
            # Generate plots
            plot_symmetry(tool_id, grid_angles, part1, part2, diff, OUTPUT_DIR)
            
            results.append({
                'Tool ID': tool_id,
                'Type': tool_type,
                'Edges': edges,
                'Condition': condition,
                'MAE': round(mae, 6),
                'RMSE': round(rmse, 6),
                'Max Diff': round(max_diff, 6),
                'Prediction': prediction
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'Tool ID': tool_id,
                'Type': tool_type,
                'Edges': edges,
                'Condition': condition,
                'MAE': None,
                'RMSE': None,
                'Max Diff': None,
                'Prediction': 'Error'
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by MAE for better readability
    results_df_sorted = results_df.sort_values('MAE', ascending=True, na_position='last')
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'symmetry_analysis_results.csv')
    results_df_sorted.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate summary plot
    valid_results = results_df_sorted[results_df_sorted['MAE'].notna()]
    if not valid_results.empty:
        plot_summary(valid_results, OUTPUT_DIR, SYMMETRY_THRESHOLD)
        print(f"Summary plots saved.")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df_sorted.to_string(index=False))
    print("=" * 70)
    
    # Save analysis metadata
    analysis_meta = {
        'method': 'Symmetry Analysis',
        'description': 'Compares two 180° segments of the tool profile',
        'threshold': SYMMETRY_THRESHOLD,
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_tools_analyzed': len(results),
        'successful_analyses': len(valid_results)
    }
    
    import json
    meta_path = os.path.join(OUTPUT_DIR, 'analysis_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(analysis_meta, f, indent=2)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
