import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

def load_and_interpolate(filepath, grid_step=1.0):
    """
    Loads the CSV data and interpolates it to a regular grid from 0 to 360 degrees.
    """
    df = pd.read_csv(filepath)
    angle_col = 'Angle (Degrees)'
    area_col = 'ROI Area (Pixels)'
    
    if angle_col not in df.columns or area_col not in df.columns:
        raise ValueError(f"Columns {angle_col} and {area_col} not found in {filepath}")
    
    angles = df[angle_col].values
    areas = df[area_col].values
    
    sorted_indices = np.argsort(angles)
    angles = angles[sorted_indices]
    areas = areas[sorted_indices]
    
    f = interp1d(angles, areas, kind='linear', fill_value="extrapolate")
    grid_angles = np.arange(0, 360, grid_step)
    interpolated_areas = f(grid_angles)
    
    return grid_angles, interpolated_areas

def sinusoidal(x_rad, amplitude, frequency, phase, offset):
    """
    Sinusoidal model function.
    x_rad: input in radians
    """
    return amplitude * np.sin(x_rad * frequency + phase) + offset

def fit_sinusoid(angles, values):
    """
    Fits a sinusoidal curve to the data.
    """
    x_rad = np.radians(angles)
    
    # Initial Guesses
    guess_amp = (np.max(values) - np.min(values)) / 2
    guess_offset = np.mean(values)
    guess_freq = 2.0 
    guess_phase = 0
    
    p0 = [guess_amp, guess_freq, guess_phase, guess_offset]
    
    try:
        popt, _ = curve_fit(sinusoidal, x_rad, values, p0=p0, maxfev=10000)
        fitted_values = sinusoidal(x_rad, *popt)
        return popt, fitted_values
    except Exception as e:
        print(f"Fitting error: {e}")
        return None, None

def plot_comparison(tool_name, grid_angles, fit1, fit2, diff, output_dir, threshold):
    """
    Generates publication-quality plots (EPS and SVG) comparing the two fitted functions.
    """
    # Global plot settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Overlay of Fitted Curves
    ax1.plot(grid_angles, fit1, '-', color='blue', linewidth=2, label='Segment 1 Fit')
    ax1.plot(grid_angles, fit2, '--', color='red', linewidth=2, label='Segment 2 Fit')
    
    ax1.set_title(f'{tool_name}: Sinusoidal Fit Comparison')
    ax1.set_ylabel('ROI Area (Pixels)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference between Fits
    ax2.plot(grid_angles, diff, color='purple', linewidth=2, label='Difference between Fits')
    
    ax2.set_title('Difference between Fitted Functions')
    ax2.set_ylabel('Difference Magnitude')
    ax2.set_xlabel('Relative Angle (Degrees)')
    ax2.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_name = tool_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Save EPS
    path_eps = os.path.join(output_dir, f'{safe_name}_fit_comparison.eps')
    plt.savefig(path_eps, format='eps', dpi=300)
    print(f"Plot saved to {path_eps}")
    
    # Save SVG
    path_svg = os.path.join(output_dir, f'{safe_name}_fit_comparison.svg')
    plt.savefig(path_svg, format='svg', dpi=300)
    print(f"Plot saved to {path_svg}")
    
    plt.close()

def plot_summary_distribution(results_df, output_dir, threshold):
    """
    Plots the distribution of Max Difference to prove the method's validity.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['red' if 'Damaged' in status else 'green' for status in results_df['Overall Status']]
    
    bars = plt.bar(results_df['Tool'], results_df['Max Difference'], color=colors, alpha=0.7)
    
    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    plt.title('Empirical Evidence: Fit Comparison Difference')
    plt.ylabel('Maximum Difference between Fits')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    path_eps = os.path.join(output_dir, 'fit_comparison_proof.eps')
    plt.savefig(path_eps, format='eps', dpi=300)
    
    path_svg = os.path.join(output_dir, 'fit_comparison_proof.svg')
    plt.savefig(path_svg, format='svg', dpi=300)
    print(f"Proof plots saved to {output_dir}")
    plt.close()

def main():
    base_dir = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA\1d_profiles"
    output_dir = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\Tool_Condition_Monitoring\symmetry_analysis\results_sinusoidal_comparison_method"
    
    # Threshold for Max Difference between the two fitted curves
    # This needs to be tuned. Since we are comparing two smooth curves, 
    # the noise is removed, so the difference should be cleaner.
    # Let's start with a conservative estimate similar to the symmetry method.
    FIT_DIFF_THRESHOLD = 0.05 
    
    tools = [
        {'name': 'Tool 074 (Damaged)', 'file': 'tool074_processed_data.csv'},
        {'name': 'Tool 114 (Good)', 'file': 'tool114_processed_data.csv'},
        {'name': 'Tool 067 (Good)', 'file': 'tool067_processed_data.csv'},
        {'name': 'Tool 070 (Damaged)', 'file': 'tool070_processed_data.csv'}
    ]
    
    summary_results = []
    
    print("Starting Sinusoidal Comparison Analysis...")
    print("-" * 50)
    
    for tool in tools:
        filepath = os.path.join(base_dir, tool['file'])
        print(f"Processing {tool['name']}...")
        
        try:
            grid_angles, values = load_and_interpolate(filepath)
            
            # Split into 2 segments
            mid = len(grid_angles) // 2
            
            # Segment 1 (0-180)
            seg1_angles = grid_angles[:mid]
            seg1_values = values[:mid]
            
            # Segment 2 (180-360)
            seg2_angles = grid_angles[mid:]
            seg2_values = values[mid:]
            
            # Fit to Segment 1
            local_angles_1 = seg1_angles - seg1_angles[0]
            popt1, fitted_values_1 = fit_sinusoid(local_angles_1, seg1_values)
            
            # Fit to Segment 2
            local_angles_2 = seg2_angles - seg2_angles[0]
            popt2, fitted_values_2 = fit_sinusoid(local_angles_2, seg2_values)
            
            if popt1 is None or popt2 is None:
                print(f"  Fit failed for one of the segments")
                continue
            
            # Compare the two fitted functions on the same domain (0-180)
            # We use the fitted parameters to generate curves on the same x-axis
            comparison_domain = local_angles_1
            curve1 = sinusoidal(np.radians(comparison_domain), *popt1)
            curve2 = sinusoidal(np.radians(comparison_domain), *popt2)
            
            diff = np.abs(curve1 - curve2)
            max_diff = np.max(diff)
            
            print(f"  Max Difference between Fits: {max_diff:.4f}")
            
            # Classification
            status = "Damaged" if max_diff > FIT_DIFF_THRESHOLD else "Good"
            print(f"  -> Prediction: {status}")
            
            summary_results.append({
                'Tool': tool['name'],
                'Max Difference': max_diff,
                'Overall Status': status
            })
            
            # Generate Plots
            plot_comparison(tool['name'], comparison_domain, curve1, curve2, diff, output_dir, FIT_DIFF_THRESHOLD)
            
        except Exception as e:
            print(f"Error processing {tool['name']}: {e}")
            
    # Save Summary
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(output_dir, 'fit_comparison_metrics.csv')
    df.to_csv(csv_path, index=False)
    print("-" * 50)
    print(f"Summary saved to {csv_path}")
    print(df)
    
    # Generate Proof Plot
    plot_summary_distribution(df, output_dir, FIT_DIFF_THRESHOLD)

if __name__ == "__main__":
    main()
