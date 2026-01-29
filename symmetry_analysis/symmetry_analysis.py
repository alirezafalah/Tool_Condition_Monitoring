import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

def load_and_interpolate(filepath, grid_step=1.0):
    """
    Loads the CSV data and interpolates it to a regular grid from 0 to 360 degrees.
    """
    df = pd.read_csv(filepath)
    
    # Extract columns (assuming standard names based on inspection)
    angle_col = 'Angle (Degrees)'
    area_col = 'ROI Area (Pixels)'
    
    if angle_col not in df.columns or area_col not in df.columns:
        raise ValueError(f"Columns {angle_col} and {area_col} not found in {filepath}")
    
    angles = df[angle_col].values
    areas = df[area_col].values
    
    # Sort by angle just in case
    sorted_indices = np.argsort(angles)
    angles = angles[sorted_indices]
    areas = areas[sorted_indices]
    
    # Handle wrap-around for interpolation if needed, but 0-360 is usually covered.
    # If 360 is missing, we can assume it's close to 0 (periodic), but let's stick to the data range.
    
    # Create interpolation function
    # fill_value="extrapolate" might be dangerous if data is missing large chunks, 
    # but for small gaps at ends it's okay.
    f = interp1d(angles, areas, kind='linear', fill_value="extrapolate")
    
    # Create regular grid
    # We want 0 to 360. 
    # Note: 0 and 360 should be the same point physically, but in the data they might be distinct points.
    # For splitting into two 180 halves:
    # Part 1: 0 to 180 (exclusive of 180 for array indexing usually, but let's be precise)
    # If we have 360 points: 0, 1, ..., 359.
    # Part 1: 0..179. Part 2: 180..359.
    
    grid_angles = np.arange(0, 360, grid_step)
    interpolated_areas = f(grid_angles)
    
    return grid_angles, interpolated_areas

def analyze_symmetry(angles, values):
    """
    Splits the data into two 180-degree segments and compares them.
    Assumes angles are on a regular grid 0..359 (or similar).
    """
    n_points = len(values)
    mid_point = n_points // 2
    
    part1 = values[:mid_point]
    part2 = values[mid_point:]
    
    # Ensure equal length (if odd number of points, trim or handle)
    min_len = min(len(part1), len(part2))
    part1 = part1[:min_len]
    part2 = part2[:min_len]
    
    # Calculate difference
    diff = np.abs(part1 - part2)
    
    # Metrics
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(diff)
    
    return part1, part2, diff, mae, rmse, max_diff

def plot_symmetry(tool_name, grid_angles, part1, part2, diff, output_dir):
    """
    Plots the comparison and difference.
    """
    # Set global font sizes for publication quality
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
    # Angles for the plot (0 to 180)
    plot_angles = grid_angles[:n_points]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Overlay
    ax1.plot(plot_angles, part1, label='0-180° Segment', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(plot_angles, part2, label='180-360° Segment', color='red', alpha=0.7, linestyle='--', linewidth=2)
    ax1.set_title(f'{tool_name} Symmetry Analysis\nOverlay of 180° Segments')
    ax1.set_ylabel('ROI Area (Pixels)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference
    ax2.plot(plot_angles, diff, label='Absolute Difference', color='purple', linewidth=2)
    ax2.set_title('Difference (Absolute Error)')
    ax2.set_xlabel('Angle (Degrees within segment)')
    ax2.set_ylabel('Difference')
    ax2.fill_between(plot_angles, diff, color='purple', alpha=0.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path_eps = os.path.join(output_dir, f'{tool_name}_symmetry_plot.eps')
    plt.savefig(output_path_eps, format='eps', dpi=300)
    print(f"Plot saved to {output_path_eps}")

    output_path_svg = os.path.join(output_dir, f'{tool_name}_symmetry_plot.svg')
    plt.savefig(output_path_svg, format='svg', dpi=300)
    print(f"Plot saved to {output_path_svg}")
    # plt.show() # Commented out for batch processing/headless env

def plot_metrics_distribution(results_df, output_dir, threshold=0.035):
    """
    Plots the MAE metrics for all tools to empirically demonstrate the separation
    between Good and Damaged tools.
    """
    plt.figure(figsize=(10, 6))
    
    # Create colors based on the label in the name for visualization
    colors = ['red' if 'Damaged' in name else 'green' for name in results_df['Tool']]
    
    bars = plt.bar(results_df['Tool'], results_df['MAE'], color=colors, alpha=0.7)
    
    # Add threshold line
    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Decision Threshold ({threshold})')
    
    plt.title('Empirical Evidence: Symmetry Error (MAE) by Tool')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    output_path_eps = os.path.join(output_dir, 'symmetry_metrics_proof.eps')
    plt.savefig(output_path_eps, format='eps', dpi=300)
    print(f"Proof plot saved to {output_path_eps}")

    output_path_svg = os.path.join(output_dir, 'symmetry_metrics_proof.svg')
    plt.savefig(output_path_svg, format='svg', dpi=300)
    print(f"Proof plot saved to {output_path_svg}")

def main():
    base_dir = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA\1d_profiles"
    output_dir = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\Tool_Condition_Monitoring\symmetry_analysis\results_symmetry_method"
    
    # Threshold determined empirically from the gap between Good (~0.018) and Damaged (~0.053)
    # A safe value is around 0.035
    SYMMETRY_THRESHOLD = 0.035
    
    tools = [
        {'name': 'Tool 074 (Damaged)', 'file': 'tool074_processed_data.csv'},
        {'name': 'Tool 114 (Good)', 'file': 'tool114_processed_data.csv'},
        {'name': 'Tool 067 (Good)', 'file': 'tool067_processed_data.csv'},
        {'name': 'Tool 070 (Damaged)', 'file': 'tool070_processed_data.csv'}
    ]
    
    results = []
    
    for tool in tools:
        filepath = os.path.join(base_dir, tool['file'])
        print(f"Processing {tool['name']}...")
        
        try:
            grid_angles, values = load_and_interpolate(filepath)
            part1, part2, diff, mae, rmse, max_diff = analyze_symmetry(grid_angles, values)
            
            # Automated Classification Logic
            prediction = "Damaged" if mae > SYMMETRY_THRESHOLD else "Good"
            print(f"  MAE: {mae:.4f} -> Prediction: {prediction}")
            
            plot_symmetry(tool['name'].replace(' ', '_').replace('(', '').replace(')', ''), 
                          grid_angles, part1, part2, diff, output_dir)
            
            results.append({
                'Tool': tool['name'],
                'MAE': mae,
                'RMSE': rmse,
                'Max Diff': max_diff,
                'Prediction': prediction
            })
            
        except Exception as e:
            print(f"Error processing {tool['name']}: {e}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'symmetry_metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nMetrics saved to {results_path}")
    print(results_df)
    
    # Generate the proof plot
    plot_metrics_distribution(results_df, output_dir, threshold=SYMMETRY_THRESHOLD)

if __name__ == "__main__":
    main()
