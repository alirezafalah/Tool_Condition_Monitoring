import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_fitting(df, num_segments, degree, visualize=True):
    """
    Perform polynomial regression on a dataset divided into adjustable segments.

    Parameters:
    - df: Input DataFrame with 'Degree' and 'Sum of Pixels' columns.
    - num_segments: Number of segments to divide the data into.
    - degree: Degree of the polynomial regression.
    - visualize: Boolean to control whether to show plots.
    
    Returns:
    - coefficients: List of coefficients for each segment's polynomial fit.
    """
    segment_size = len(df) // num_segments
    coefficients = []
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']  # For different segments

    plt.figure(figsize=(16, 12))
    plt.scatter(df["Degree"], df["Sum of Pixels"], color="blue", label="Original data")

    for i in range(num_segments):
        segment = df.iloc[i * segment_size: (i + 1) * segment_size]
        X_segment = segment[['Degree']].values
        Y_segment = segment['Sum of Pixels'].values

        # Polynomial features transformation
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_segment)

        # Linear regression model fitting
        model = LinearRegression()
        model.fit(X_poly, Y_segment)

        # Predictions
        Y_pred = model.predict(X_poly)

        # Append coefficients
        coefficients.append(model.coef_)

        # Plot segment polynomial fit
        plt.plot(segment['Degree'], Y_pred, color=colors[i % len(colors)], 
                 label=f'Segment {i + 1}: {degree}-degree Polynomial Fit')

    # Add vertical segment separators
    for i in range(1, num_segments):
        plt.axvline(x=i * (360 / num_segments), color='green', linestyle='--')
    plt.axvline(x=360, color='green', linestyle='--')
    plt.xticks([i * (360 / num_segments) for i in range(num_segments + 1)],
               [str(int(i * (360 / num_segments))) for i in range(num_segments + 1)], color='green')

    plt.xlabel('Degree')
    plt.ylabel('Sum of Pixels')
    plt.title(f'Polynomial regression with {num_segments} Segments')
    plt.legend()
    plt.grid(True)

    if visualize:
        plt.show()

    return coefficients


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
    drill_coefficients = polynomial_fitting(df_drill, num_segments=2, degree=5)

    # Print coefficients for inspection
    print("\nChamfer Coefficients:", chamfer_coefficients)
    print("\nDrill Coefficients:", drill_coefficients)
