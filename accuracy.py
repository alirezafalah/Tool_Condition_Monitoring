import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from savitzky_peak_segmentation import savgol_peak_finder
from scipy.stats import pearsonr

def shift_data_to_breaking_point(data_path):
    # Get the smoothed data and pattern boundaries
    smoothed_pixels, pattern_boundaries = savgol_peak_finder(data_path)
    
    # Load the original data
    tool_data = pd.read_csv(data_path)
    
    # Extract the first breaking point (pattern start index)
    first_pattern_start, _ = pattern_boundaries[0]
    
    # Shift the data so that it starts at the first breaking point
    shifted_smoothed_pixels = np.roll(smoothed_pixels, -first_pattern_start)
    shifted_degrees = np.roll(tool_data['Degree'].values, -first_pattern_start)
    shifted_original_pixels = np.roll(tool_data['Sum of Pixels'].values, -first_pattern_start)
    
    # Create a DataFrame with the shifted data
    shifted_data = pd.DataFrame({
        'Degree': shifted_degrees,
        'Sum of Pixels': shifted_original_pixels,
        'Smoothed Pixels': shifted_smoothed_pixels
    })
    
    # Adjust the Degree values to start from zero and wrap around 360 degrees
    degree_shift = shifted_data['Degree'][0]
    shifted_data['Degree'] = (shifted_data['Degree'] - degree_shift) % 360  # Assuming degrees are in [0, 360)
    
    # Adjust the pattern boundaries after shifting
    adjusted_pattern_boundaries = []
    total_length = len(shifted_data)
    for start, end in pattern_boundaries:
        adjusted_start = (start - first_pattern_start) % total_length
        adjusted_end = (end - first_pattern_start) % total_length
        adjusted_pattern_boundaries.append((adjusted_start, adjusted_end))
    
    # Initialize lists to store coefficients and fitted curves
    segment_coeffs = []
    segment_fitted_curves = []
    segment_degrees_list = []
    
    # Plot the shifted data with segments highlighted
    plt.figure(figsize=(12, 7))
    plt.plot(shifted_data['Degree'], shifted_data['Smoothed Pixels'], label="Shifted Smoothed Sum of Pixels", color='black')
    
    colors = ['green', 'red', 'blue', 'purple', 'yellow', 'cyan', 'magenta', 'orange']
    residual_errors = []
    
    for i, (start, end) in enumerate(adjusted_pattern_boundaries):
        color = colors[i % len(colors)]
        
        # Handle wrap-around segments
        if start < end:
            segment_indices = np.arange(int(start), int(end))
        else:
            # Wrap-around segment
            segment_indices = np.concatenate((np.arange(int(start), total_length), np.arange(0, int(end))))
        
        segment_degrees = shifted_data['Degree'].iloc[segment_indices].values
        segment_pixels = shifted_data['Smoothed Pixels'].iloc[segment_indices].values
        
        # Store degrees for later comparison
        segment_degrees_list.append(segment_degrees)
        
        # Plot the segment background
        plt.fill_between(segment_degrees, segment_pixels.min(), segment_pixels.max(), color=color, alpha=0.1)
        
        # Fit a polynomial curve to the segment
        # You can adjust the polynomial degree as needed
        poly_degree = 3
        coeffs = np.polyfit(segment_degrees, segment_pixels, deg=poly_degree)
        poly_fit = np.poly1d(coeffs)
        fitted_pixels = poly_fit(segment_degrees)
        
        # Store coefficients and fitted curves
        segment_coeffs.append(coeffs)
        segment_fitted_curves.append(fitted_pixels)
        
        # Plot the fitted curve
        plt.plot(segment_degrees, fitted_pixels, color=color, linestyle='--', label=f'Fitted Curve Segment {i+1}')
        
        # Calculate residual error (sum of squared residuals)
        residual = np.sum((segment_pixels - fitted_pixels) ** 2)
        residual_errors.append(residual)
        
    plt.title("Shifted Data Starting at First Breaking Point with Fitted Curves")
    plt.xlabel("Degree")
    plt.ylabel("Sum of Pixels")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Print residual errors for each segment
    for i, residual in enumerate(residual_errors):
        print(f"Residual Error for Segment {i+1}: {residual:.4f}")
    
    # Compare the fitted curves
    compare_fitted_curves(segment_coeffs, segment_fitted_curves, segment_degrees_list)
    
    return shifted_data, adjusted_pattern_boundaries, residual_errors

def compare_fitted_curves(segment_coeffs, segment_fitted_curves, segment_degrees_list):
    num_segments = len(segment_coeffs)
    
    # Initialize lists to store comparison metrics
    coeff_differences = []
    mse_values = []
    correlation_coefficients = []
    
    # Define a common normalized degree range
    normalized_degrees = np.linspace(0, 1, 100)
    
    # For each segment, evaluate the fitted curve over the normalized degrees
    normalized_fitted_curves = []
    
    for i in range(num_segments):
        segment_degrees = segment_degrees_list[i]
        min_degree = segment_degrees.min()
        max_degree = segment_degrees.max()
        # Normalize the degrees to [0, 1]
        segment_degrees_normalized = (segment_degrees - min_degree) / (max_degree - min_degree)
        # Evaluate the fitted polynomial over the normalized degrees
        poly_fit = np.poly1d(segment_coeffs[i])
        # Interpolate the fitted curve to the common normalized degrees
        fitted_curve_normalized = poly_fit(segment_degrees_normalized)
        fitted_curve_common = np.interp(normalized_degrees, segment_degrees_normalized, fitted_curve_normalized)
        normalized_fitted_curves.append(fitted_curve_common)
    
    # Now compare each segment's normalized fitted curve to the reference
    reference_curve = normalized_fitted_curves[0]
    
    for i in range(1, num_segments):
        # Coefficient difference
        coeff_diff = np.abs(segment_coeffs[0] - segment_coeffs[i])
        coeff_differences.append(coeff_diff)
        
        # Mean Squared Error
        mse = np.mean((reference_curve - normalized_fitted_curves[i]) ** 2)
        mse_values.append(mse)
        
        # Correlation Coefficient
        corr_coef, _ = pearsonr(reference_curve, normalized_fitted_curves[i])
        correlation_coefficients.append(corr_coef)
        
        # Print comparison results
        print(f"\nComparison between Segment 1 and Segment {i+1}:")
        print(f"Coefficient Differences: {coeff_diff}")
        print(f"MSE between fitted curves: {mse:.4f}")
        print(f"Correlation Coefficient: {corr_coef:.4f}")
        
    # Optionally, you can store or return these comparison metrics for further analysis

# Run the function
shift_data_to_breaking_point("data/table(intact).csv")
