"""
Sinusoidal Comparison Analysis Method for Tool Condition Monitoring
====================================================================
This script fits independent sinusoidal curves to each 180° segment,
then compares the fitted curves to detect asymmetry.

Output: DATA/threshold_analysis/sinusoidal_comparison_method/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
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
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", "subtracting_fitted_functions")

# Threshold for classification (can be tuned based on probability analysis)
SINUSOIDAL_THRESHOLD = 0.05

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

def sinusoid(x, A, omega, phi, offset):
    """Sinusoidal function for fitting."""
    return A * np.sin(omega * x + phi) + offset

def fit_sinusoid(angles, values, num_edges=2):
    """Fits a sinusoidal curve to the data."""
    angles_rad = np.deg2rad(angles)
    
    A_init = (np.max(values) - np.min(values)) / 2
    omega_init = float(num_edges)
    phi_init = 0.0
    offset_init = np.mean(values)
    
    p0 = [A_init, omega_init, phi_init, offset_init]
    
    bounds = (
        [0, num_edges - 0.5, -2 * np.pi, np.min(values) - 1e6],
        [np.inf, num_edges + 0.5, 2 * np.pi, np.max(values) + 1e6]
    )
    
    try:
        popt, pcov = curve_fit(sinusoid, angles_rad, values, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception as e:
        print(f"    Fitting failed: {e}")
        return None

def analyze_sinusoidal_comparison(grid_angles, values, num_edges=2):
    """
    Fits independent sinusoids to each 180° segment and compares the fitted curves.
    """
    n_points = len(values)
    mid_point = n_points // 2
    
    part1_angles = grid_angles[:mid_point]
    part1_values = values[:mid_point]
    
    part2_angles = grid_angles[mid_point:mid_point + len(part1_angles)]
    part2_values = values[mid_point:mid_point + len(part1_values)]
    
    min_len = min(len(part1_angles), len(part2_angles))
    part1_angles = part1_angles[:min_len]
    part1_values = part1_values[:min_len]
    part2_angles = part2_angles[:min_len]
    part2_values = part2_values[:min_len]
    
    # Normalize part2 angles to same range as part1 for comparison
    part2_angles_shifted = part1_angles.copy()
    
    # Fit sinusoids
    params1 = fit_sinusoid(part1_angles, part1_values, num_edges=1)
    params2 = fit_sinusoid(part2_angles_shifted, part2_values, num_edges=1)
    
    if params1 is None or params2 is None:
        return None
    
    # Generate fitted curves on the same angle grid
    common_angles = part1_angles
    common_angles_rad = np.deg2rad(common_angles)
    
    fitted1 = sinusoid(common_angles_rad, *params1)
    fitted2 = sinusoid(common_angles_rad, *params2)
    
    # Calculate metrics between the two fitted curves
    diff = np.abs(fitted1 - fitted2)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(diff)
    
    # Normalize MAE by amplitude for scale-invariance
    amplitude1 = params1[0]
    amplitude2 = params2[0]
    avg_amplitude = (amplitude1 + amplitude2) / 2
    normalized_mae = mae / avg_amplitude if avg_amplitude > 0 else mae
    
    return {
        'part1_angles': part1_angles,
        'part1_values': part1_values,
        'part2_angles_shifted': part2_angles_shifted,
        'part2_values': part2_values,
        'common_angles': common_angles,
        'fitted1': fitted1,
        'fitted2': fitted2,
        'params1': params1,
        'params2': params2,
        'diff': diff,
        'mae': mae,
        'rmse': rmse,
        'max_diff': max_diff,
        'normalized_mae': normalized_mae
    }

def plot_sinusoidal_comparison(tool_id, analysis_result, output_dir):
    """Plots the sinusoidal comparison analysis."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Segment 1 with fit
    ax1 = axes[0, 0]
    ax1.scatter(analysis_result['part1_angles'], analysis_result['part1_values'], 
               alpha=0.4, s=20, label='Data (0-180°)', color='blue')
    ax1.plot(analysis_result['common_angles'], analysis_result['fitted1'], 
            color='blue', linewidth=2, label='Sinusoidal Fit')
    ax1.set_title(f'{tool_id}\nSegment 1 (0-180°) with Sinusoidal Fit')
    ax1.set_xlabel('Angle (Degrees)')
    ax1.set_ylabel('ROI Area (Pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Segment 2 with fit
    ax2 = axes[0, 1]
    ax2.scatter(analysis_result['part2_angles_shifted'], analysis_result['part2_values'], 
               alpha=0.4, s=20, label='Data (180-360°)', color='red')
    ax2.plot(analysis_result['common_angles'], analysis_result['fitted2'], 
            color='red', linewidth=2, label='Sinusoidal Fit')
    ax2.set_title(f'Segment 2 (180-360°) with Sinusoidal Fit')
    ax2.set_xlabel('Angle (Degrees, shifted)')
    ax2.set_ylabel('ROI Area (Pixels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Overlay of fitted curves
    ax3 = axes[1, 0]
    ax3.plot(analysis_result['common_angles'], analysis_result['fitted1'], 
            color='blue', linewidth=2, label='Fit: Segment 1', alpha=0.8)
    ax3.plot(analysis_result['common_angles'], analysis_result['fitted2'], 
            color='red', linewidth=2, linestyle='--', label='Fit: Segment 2', alpha=0.8)
    ax3.set_title('Comparison of Fitted Sinusoidal Curves')
    ax3.set_xlabel('Angle (Degrees)')
    ax3.set_ylabel('Fitted ROI Area')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Difference between fitted curves
    ax4 = axes[1, 1]
    ax4.plot(analysis_result['common_angles'], analysis_result['diff'], 
            color='purple', linewidth=2)
    ax4.fill_between(analysis_result['common_angles'], analysis_result['diff'], 
                     color='purple', alpha=0.2)
    ax4.set_title(f"Difference Between Fitted Curves\nMAE: {analysis_result['mae']:.4f}, Normalized: {analysis_result['normalized_mae']:.4f}")
    ax4.set_xlabel('Angle (Degrees)')
    ax4.set_ylabel('Absolute Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_sinusoidal_comparison.{fmt}')
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
    
    bars = ax.bar(results_df['Tool ID'], results_df['Normalized MAE'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    ax.set_title('Sinusoidal Comparison Analysis: Normalized MAE by Tool')
    ax.set_ylabel('Normalized Mean Absolute Error')
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
        path = os.path.join(output_dir, f'sinusoidal_comparison_summary.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SINUSOIDAL COMPARISON ANALYSIS FOR TOOL CONDITION MONITORING")
    print("=" * 70)
    print(f"Profiles Directory: {PROFILES_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Threshold (Normalized MAE): {SINUSOIDAL_THRESHOLD}")
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
        
        try:
            num_edges = int(edges) if edges != 'N/A' else 2
        except:
            num_edges = 2
        
        print(f"Processing {tool_id} (Type: {tool_type}, Edges: {edges}, Condition: {condition})...")
        
        try:
            grid_angles, values = load_and_interpolate(filepath)
            result = analyze_sinusoidal_comparison(grid_angles, values, num_edges=num_edges)
            
            if result is None:
                raise ValueError("Fitting failed for one or both segments")
            
            # Classification based on normalized MAE
            prediction = "Damaged" if result['normalized_mae'] > SINUSOIDAL_THRESHOLD else "Good"
            
            print(f"  Normalized MAE: {result['normalized_mae']:.4f} -> Prediction: {prediction}")
            
            # Generate plots
            plot_sinusoidal_comparison(tool_id, result, OUTPUT_DIR)
            
            results.append({
                'Tool ID': tool_id,
                'Type': tool_type,
                'Edges': edges,
                'Condition': condition,
                'MAE': round(result['mae'], 6),
                'RMSE': round(result['rmse'], 6),
                'Max Diff': round(result['max_diff'], 6),
                'Normalized MAE': round(result['normalized_mae'], 6),
                'Amplitude 1': round(result['params1'][0], 4),
                'Amplitude 2': round(result['params2'][0], 4),
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
                'Normalized MAE': None,
                'Amplitude 1': None,
                'Amplitude 2': None,
                'Prediction': 'Error'
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Normalized MAE for better readability
    results_df_sorted = results_df.sort_values('Normalized MAE', ascending=True, na_position='last')
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'sinusoidal_comparison_results.csv')
    results_df_sorted.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate summary plot
    valid_results = results_df_sorted[results_df_sorted['Normalized MAE'].notna()]
    if not valid_results.empty:
        plot_summary(valid_results, OUTPUT_DIR, SINUSOIDAL_THRESHOLD)
        print(f"Summary plots saved.")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df_sorted.to_string(index=False))
    print("=" * 70)
    
    # Save analysis metadata
    analysis_meta = {
        'method': 'Sinusoidal Comparison Analysis',
        'description': 'Fits independent sinusoids to each 180° segment and compares the fitted curves',
        'threshold': SINUSOIDAL_THRESHOLD,
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
