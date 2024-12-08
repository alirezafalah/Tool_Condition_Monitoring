import numpy as np
import matplotlib.pyplot as plt

def compare_polynomial_segments(segment_coeffs, ref_segment_coeffs, domain, segment_labels):
    """
    Compare polynomial segments with reference segments and compute errors.

    Parameters:
    - segment_coeffs: List of coefficients for the polynomials to compare.
    - ref_segment_coeffs: List of reference coefficients for comparison.
    - domain: Array of degrees over which to evaluate.
    - segment_labels: Labels for each segment (e.g., ['Green', 'Blue', ...]).

    Returns:
    - errors: List of percentage errors for each segment.
    """
    plt.figure(figsize=(12, 7))
    errors = []

    # Loop through all segments
    for i, (coeffs, ref_coeffs) in enumerate(zip(segment_coeffs, ref_segment_coeffs)):
        # Evaluate the polynomials
        poly = np.poly1d(coeffs)
        ref_poly = np.poly1d(ref_coeffs)

        # Evaluate the fitted and reference curves
        fitted_pixels = poly(domain)
        ref_fitted_pixels = ref_poly(domain)

        # Compute the difference and errors
        difference = fitted_pixels - ref_fitted_pixels
        error_norm = np.linalg.norm(difference, ord=2)
        reference_norm = np.linalg.norm(ref_fitted_pixels, ord=2)
        percentage_error = (error_norm / reference_norm) * 100
        errors.append(percentage_error)

        # Plot the fitted and reference curves
        plt.plot(domain, fitted_pixels, label=f"Segment {segment_labels[i]} (Fitted)", linestyle='--')
        plt.plot(domain, ref_fitted_pixels, label=f"Segment {segment_labels[i]} (Reference)", linestyle='-')
        plt.fill_between(domain, fitted_pixels, ref_fitted_pixels, alpha=0.2, label=f"Error {segment_labels[i]}: {percentage_error:.2f}%")

        # Annotate the error on the plot
        mid_point = domain[len(domain) // 2]
        plt.text(mid_point, max(fitted_pixels.max(), ref_fitted_pixels.max()), f"{percentage_error:.2f}%", ha='center')

    plt.title("Comparison of Polynomial Segments")
    plt.xlabel("Degree")
    plt.ylabel("Fitted Pixels")
    plt.legend()
    plt.grid()
    plt.show()

    return errors


# Define the coefficients for the intact and broken sets
intact_coeffs = [
    [-0.0579, 0.0565, -0.00099, 0.000],  # base
    [-0.043, 0.045, -0.00072, 0.000],    # poly1
    [-0.0427, 0.0516, -0.00087, 0.00],   # poly2
    [-0.0809, 0.056, -0.0098, 0.00]      # poly3
]

broken_coeffs = [
    [0.3603, 0.0335, -0.00063, 0.00],    # base
    [0.0872, 0.0154, -0.00023, 0.00],    # poly1
    [0.3134, 0.0404, -0.00079, 0.00],    # poly2
    [0.3213, 0.0412, -0.00081, 0.00]     # poly3
]

# Define the domain for evaluation
domain = np.linspace(1, 90, 90)  # Domain from 1 to 90

# Segment labels for plotting
labels = ["Base", "Poly1", "Poly2", "Poly3"]

# Compare intact set with itself (to simulate a reference comparison)
print("Comparing Intact Set:")
intact_errors = compare_polynomial_segments(intact_coeffs, intact_coeffs, domain, labels)

# Compare broken set to intact set
print("Comparing Broken Set to Intact Set:")
broken_errors = compare_polynomial_segments(broken_coeffs, intact_coeffs, domain, labels)

# Print percentage errors
print("Intact Set Errors:", intact_errors)
print("Broken Set Errors:", broken_errors)


### Here was another method!

# import numpy as np
# import pandas as pd

# # INTACT
# base_coeffs = [-0.0579, 0.0565, -0.00099, 0.000]
# poly1_coeffs = [-0.043, 0.045, -0.00072, 0.000]
# poly2_coeffs = [-0.0427, 0.0516, -0.00087, 0.00]
# poly3_coeffs = [-0.0809, 0.056, -0.0098, 0.00]

# # BROKEN
# # base_coeffs =  [0.0872, 0.0154, -0.00023, 0.00] 
# # poly1_coeffs =  [0.3603, 0.0335, -0.00063, 0.00]
# # poly2_coeffs = [0.3134, 0.0404, -0.00079, 0.00]  
# # poly3_coeffs = [0.3213, 0.0412, -0.00081, 0.00] 

# # Extract only a and b coefficients
# base_ab = base_coeffs[:2]
# poly1_ab = poly1_coeffs[:2]
# poly2_ab = poly2_coeffs[:2]
# poly3_ab = poly3_coeffs[:2]

# # Function to calculate percentage difference
# def percentage_diff(base, other):
#     return [abs((b - o) / b) * 100 for b, o in zip(base, other)]

# # Calculate percentage differences
# poly1_diff = percentage_diff(base_ab, poly1_ab)
# poly2_diff = percentage_diff(base_ab, poly2_ab)
# poly3_diff = percentage_diff(base_ab, poly3_ab)

# # Aggregate results in a DataFrame for clarity
# results = pd.DataFrame({
#     "Polynomial": ["Poly1", "Poly2", "Poly3"],
#     "A Difference (%)": [poly1_diff[0], poly2_diff[0], poly3_diff[0]],
#     "B Difference (%)": [poly1_diff[1], poly2_diff[1], poly3_diff[1]],
#     "Average Difference (%)": [
#         sum(poly1_diff) / 2,
#         sum(poly2_diff) / 2,
#         sum(poly3_diff) / 2
#     ]
# })

# print(results)
