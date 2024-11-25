import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def polynomial_fitting_iterative(df, num_segments, max_degree, visualize=True):
    """
    Perform polynomial regression iteratively for degrees 2 to max_degree for each segment
    and determine the degree with the highest R2.

    Parameters:
    - df: Input DataFrame with 'Degree' and 'Sum of Pixels' columns.
    - num_segments: Number of segments to divide the data into.
    - max_degree: Maximum polynomial degree to test.
    - visualize: Boolean to control whether to show plots.

    Returns:
    - best_fits: List of dictionaries for each segment with the best degree, coefficients, and R2.
    - shifted_segments: List of DataFrames, each representing a shifted segment.
    """
    segment_size = len(df) // num_segments
    best_fits = []
    shifted_segments = []
    colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'red']  # For different segments

    plt.figure(figsize=(16, 12))

    for i in range(num_segments):
        # Extract segment
        segment = df.iloc[i * segment_size: (i + 1) * segment_size].copy()

        # Shift the degree axis to start at 0
        segment["Degree"] = segment["Degree"] - segment["Degree"].iloc[0]

        # Store the shifted segment
        shifted_segments.append(segment)

        # Data for the current segment
        X_segment = segment[['Degree']].values
        Y_segment = segment['Sum of Pixels'].values

        # Variables to track the best polynomial fit
        best_r2 = float('-inf')
        best_degree = None
        best_coefficients = None
        best_predictions = None

        # Iterate over polynomial degrees
        for degree in range(2, max_degree + 1):
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_segment)

            # Linear regression model fitting
            model = LinearRegression()
            model.fit(X_poly, Y_segment)

            # Predictions and R2 calculation
            Y_pred = model.predict(X_poly)
            r2 = r2_score(Y_segment, Y_pred)

            # Check if this is the best R2 so far
            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree
                best_coefficients = [model.intercept_, *model.coef_[1:]]
                best_predictions = Y_pred

        # Append best fit for the current segment
        best_fits.append({
            "segment": i + 1,
            "best_degree": best_degree,
            "best_r2": best_r2,
            "coefficients": best_coefficients
        })

        # Plot the best polynomial fit for the segment
        plt.plot(segment['Degree'], best_predictions, color=colors[i % len(colors)],
                 label=f'Segment {i + 1}: Best Degree {best_degree} (R2={best_r2:.3f})')
        plt.scatter(segment['Degree'], segment['Sum of Pixels'], color=colors[i % len(colors)], alpha=0.6)

    plt.xlabel('Shifted Degree (per segment)')
    plt.ylabel('Sum of Pixels')
    plt.title(f'Best Polynomial Regression Fit (2 to {max_degree} Degree) for {num_segments} Segments')
    plt.legend()
    plt.grid(True)

    if visualize:
        plt.show()

    return best_fits, shifted_segments


# Example Usage:
if __name__ == "__main__":
    # Load dataset (adjust the path to your data)
    df_chamfer = pd.read_csv(r"processed_data/chamfer_processed.csv")
    df_drill = pd.read_csv(r"processed_data/drill_processed.csv")

    # Polynomial fitting for chamfer (4 segments, degrees 2 to 5)
    print("Chamfer Best Fits:")
    chamfer_best_fits, _ = polynomial_fitting_iterative(df_chamfer, num_segments=4, max_degree=10)

    # Polynomial fitting for drill (2 segments, degrees 2 to 5)
    print("\nDrill Best Fits:")
    drill_best_fits, _ = polynomial_fitting_iterative(df_drill, num_segments=2, max_degree=10)

    # Print results for inspection
    print("\nChamfer Best Fits:", chamfer_best_fits)
    print("\nDrill Best Fits:", drill_best_fits)


### A VERSION THAT IDENTIFIES THE MOST SIGNIFICANT COEEFICIENT
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# def polynomial_fitting_iterative_with_significance(df, num_segments, max_degree, visualize=True):
#     """
#     Perform polynomial regression iteratively for degrees 2 to max_degree for each segment,
#     determine the degree with the highest R2, and find the most significant coefficients.

#     Parameters:
#     - df: Input DataFrame with 'Degree' and 'Sum of Pixels' columns.
#     - num_segments: Number of segments to divide the data into.
#     - max_degree: Maximum polynomial degree to test.
#     - visualize: Boolean to control whether to show plots.

#     Returns:
#     - best_fits: List of dictionaries for each segment with the best degree, coefficients, R2, and significance.
#     - shifted_segments: List of DataFrames, each representing a shifted segment.
#     """
#     segment_size = len(df) // num_segments
#     best_fits = []
#     shifted_segments = []
#     colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'red']  # For different segments

#     plt.figure(figsize=(16, 12))

#     for i in range(num_segments):
#         # Extract segment
#         segment = df.iloc[i * segment_size: (i + 1) * segment_size].copy()

#         # Shift the degree axis to start at 0
#         segment["Degree"] = segment["Degree"] - segment["Degree"].iloc[0]

#         # Store the shifted segment
#         shifted_segments.append(segment)

#         # Data for the current segment
#         X_segment = segment[['Degree']].values
#         Y_segment = segment['Sum of Pixels'].values

#         # Variables to track the best polynomial fit
#         best_r2 = float('-inf')
#         best_degree = None
#         best_coefficients = None
#         best_predictions = None

#         # Iterate over polynomial degrees
#         for degree in range(2, max_degree + 1):
#             poly = PolynomialFeatures(degree=degree)
#             X_poly = poly.fit_transform(X_segment)

#             # Linear regression model fitting
#             model = LinearRegression()
#             model.fit(X_poly, Y_segment)

#             # Predictions and R2 calculation
#             Y_pred = model.predict(X_poly)
#             r2 = r2_score(Y_segment, Y_pred)

#             # Check if this is the best R2 so far
#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_degree = degree
#                 best_coefficients = [model.intercept_, *model.coef_[1:]]  # Skip the first coefficient
#                 best_predictions = Y_pred

#         # Identify the most significant coefficients (sorted by absolute value)
#         significant_coefficients = sorted(
#             enumerate(best_coefficients),
#             key=lambda x: abs(x[1]),
#             reverse=True
#         )
#         # Format coefficients for output
#         formatted_coefficients = [
#             {"term": f"x^{i}" if i > 0 else "intercept", "value": coef}
#             for i, coef in significant_coefficients
#         ]

#         # Append best fit for the current segment
#         best_fits.append({
#             "segment": i + 1,
#             "best_degree": best_degree,
#             "best_r2": best_r2,
#             "coefficients": best_coefficients,
#             "significant_coefficients": formatted_coefficients
#         })

#         # Plot the best polynomial fit for the segment
#         plt.plot(segment['Degree'], best_predictions, color=colors[i % len(colors)],
#                  label=f'Segment {i + 1}: Best Degree {best_degree} (R2={best_r2:.3f})')
#         plt.scatter(segment['Degree'], segment['Sum of Pixels'], color=colors[i % len(colors)], alpha=0.6)

#     plt.xlabel('Shifted Degree (per segment)')
#     plt.ylabel('Sum of Pixels')
#     plt.title(f'Best Polynomial Regression Fit (2 to {max_degree} Degree) for {num_segments} Segments')
#     plt.legend()
#     plt.grid(True)

#     if visualize:
#         plt.show()

#     return best_fits, shifted_segments


# # Example Usage:
# if __name__ == "__main__":
#     # Load dataset (adjust the path to your data)
#     df_chamfer = pd.read_csv(r"processed_data/chamfer_processed.csv")
#     df_drill = pd.read_csv(r"processed_data/drill_processed.csv")

#     # Polynomial fitting for chamfer (4 segments, degrees 2 to 5)
#     print("Chamfer Best Fits with Significance:")
#     chamfer_best_fits, _ = polynomial_fitting_iterative_with_significance(df_chamfer, num_segments=4, max_degree=5)

#     # Polynomial fitting for drill (2 segments, degrees 2 to 5)
#     print("\nDrill Best Fits with Significance:")
#     drill_best_fits, _ = polynomial_fitting_iterative_with_significance(df_drill, num_segments=2, max_degree=5)

#     # Print significant coefficients for inspection
#     for fit in chamfer_best_fits:
#         print(f"\nSegment {fit['segment']} - Best Degree: {fit['best_degree']} - R2: {fit['best_r2']:.3f}")
#         print("Significant Coefficients (sorted by magnitude):")
#         for term in fit["significant_coefficients"]:
#             print(f"  {term['term']}: {term['value']:.4f}")
