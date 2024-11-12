import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from savitzky_peak_segmentation import savgol_peak_finder
from scipy.stats import pearsonr

def shift_data_to_breaking_point(data_path, ref_coeffs_list=None, ref_segment_degrees_list=None):
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

    # Loop to fit polynomial curves for each segment
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
        poly_degree = 2 
        coeffs = np.polyfit(segment_degrees, segment_pixels, deg=poly_degree)
        poly_fit = np.poly1d(coeffs)
        fitted_pixels = poly_fit(segment_degrees)

        # Store coefficients and fitted curves
        segment_coeffs.append(coeffs)
        segment_fitted_curves.append(fitted_pixels)

        # Plot the fitted curve
        plt.plot(segment_degrees, fitted_pixels, color=color, linestyle='--', label=f'Fitted Curve Segment {i+1}')

        # If reference coefficients are provided, compare with the reference curve
        if ref_coeffs_list is not None and ref_segment_degrees_list is not None:
            # Interpolate reference curve to match current segment degrees
            ref_coeffs = ref_coeffs_list[i]
            ref_poly_fit = np.poly1d(ref_coeffs)
            ref_fitted_pixels = ref_poly_fit(segment_degrees)

            # Plot the reference curve
            plt.plot(segment_degrees, ref_fitted_pixels, color=color, linestyle='-', label=f'Reference Curve Segment {i+1}')

            # Calculate the difference
            difference = fitted_pixels - ref_fitted_pixels

            # Calculate the 2-norm error and percentage error
            error_norm = np.linalg.norm(difference, ord=2)
            reference_norm = np.linalg.norm(ref_fitted_pixels, ord=2)
            percentage_error = (error_norm / reference_norm) * 100

            # Fill the area between the curves
            plt.fill_between(segment_degrees, fitted_pixels, ref_fitted_pixels, color=color, alpha=0.3)

            # Display the percentage error on the plot
            mid_degree = np.median(segment_degrees)
            max_pixel = max(fitted_pixels.max(), ref_fitted_pixels.max())
            plt.text(mid_degree, max_pixel, f'{percentage_error:.2f}%', color=color, fontsize=10, ha='center')

            # Print out the results
            print(f"Segment {i+1}:")
            print(f"  2-Norm Error: {error_norm:.4f}")
            print(f"  Percentage Error: {percentage_error:.2f}%")
        else:
            # If no reference, use the first segment as the reference for error calculation
            if i == 0:
                reference_curve = fitted_pixels
                reference_norm = np.linalg.norm(reference_curve, ord=2)
            else:
                # Compute the error as the difference from the reference curve
                error_vector = fitted_pixels - reference_curve[:len(fitted_pixels)]

                # Calculate the 2-norm of the error
                error_norm = np.linalg.norm(error_vector, ord=2)

                # Calculate the percentage error relative to the reference curve
                percentage_error = (error_norm / reference_norm) * 100

                # Print out the results
                print(f"Segment {i+1} vs Reference Segment:")
                print(f"  2-Norm Error: {error_norm:.4f}")
                print(f"  Percentage Error: {percentage_error:.2f}%")

    plt.title(f"Comparision of Fitted Curves with Reference")
    plt.xlabel("Degree")
    plt.ylabel("Sum of Pixels")
    plt.grid(True)
    plt.legend()
    plt.show()

    return shifted_data, adjusted_pattern_boundaries, segment_coeffs, segment_degrees_list

# First, process the reference CSV (intact.csv) to get the reference curves and coefficients
print("Processing reference data (intact.csv)...")
ref_shifted_data, ref_adjusted_pattern_boundaries, ref_segment_coeffs, ref_segment_degrees_list = shift_data_to_breaking_point("data/intact.csv")

# Now, process the new CSV and compare it to the reference
print("\nProcessing new data (new_data.csv) and comparing to reference...")
new_shifted_data, new_adjusted_pattern_boundaries, _, _ = shift_data_to_breaking_point(
    "data/broken.csv", 
    ref_coeffs_list=ref_segment_coeffs, 
    ref_segment_degrees_list=ref_segment_degrees_list
)
