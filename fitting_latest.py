import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_fitting(df, num_segments, degree, visualize=True):
    """
    Perform polynomial regression on a dataset with each segment shifted to start from 0 degrees.

    Parameters:
    - df: Input DataFrame with 'Degree' and 'Sum of Pixels' columns.
    - num_segments: Number of segments to divide the data into.
    - degree: Degree of the polynomial regression.
    - visualize: Boolean to control whether to show plots.
    
    Returns:
    - coefficients: List of coefficients for each segment's polynomial fit.
    - shifted_segments: List of DataFrames, each representing a shifted segment.
    """
    segment_size = len(df) // num_segments
    coefficients = []
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

        # Polynomial features transformation
        X_segment = segment[['Degree']].values
        Y_segment = segment['Sum of Pixels'].values
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_segment)

        # Linear regression model fitting
        model = LinearRegression()
        model.fit(X_poly, Y_segment)

        # Predictions
        Y_pred = model.predict(X_poly)

        # Append coefficients
        coefficients.append([model.intercept_, *model.coef_[1:]])

        # Plot segment polynomial fit
        plt.plot(segment['Degree'], Y_pred, color=colors[i % len(colors)], 
                 label=f'Shifted Segment {i + 1}: {degree}-degree Polynomial Fit')
        plt.scatter(segment['Degree'], segment['Sum of Pixels'], color=colors[i % len(colors)], alpha=0.6)

    plt.xlabel('Shifted Degree (per segment)')
    plt.ylabel('Sum of Pixels')
    plt.title(f'Polynomial regression with {num_segments} Shifted Segments')
    plt.legend()
    plt.grid(True)

    if visualize:
        plt.show()

    return coefficients, shifted_segments

# Example Usage:
if __name__ == "__main__":
    # Load dataset (adjust the path to your data)
    df_chamfer = pd.read_csv(r"processed_data/chamfer_processed.csv")
    df_drill = pd.read_csv(r"processed_data/drill_processed.csv")

    # Polynomial fitting for chamfer (4 segments)
    print("Chamfer Polynomial Coefficients:")
    chamfer_coefficients = polynomial_fitting(df_chamfer, num_segments=4, degree=2)

    # Polynomial fitting for drill (2 segments)
    print("\nDrill Polynomial Coefficients:")
    drill_coefficients = polynomial_fitting(df_drill, num_segments=2, degree=2)

    # Print coefficients for inspection
    print("\nChamfer Coefficients:", chamfer_coefficients)
    print("\nDrill Coefficients:", drill_coefficients)
