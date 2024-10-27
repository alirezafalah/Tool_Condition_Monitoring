import pandas as pd
import numpy as np
from savitzky_peak_segmentation import savgol_peak_finder

import matplotlib.pyplot as plt

# File path for the intact tool data
file_path = r"data\table(intact).csv"

def fit_polynomials_to_patterns(data_path):
    # Get the pattern boundaries from the savgol_peak_finder function
    tool_data = pd.read_csv(data_path)
    _, pattern_boundaries = savgol_peak_finder(data_path)

    # Fit a second-degree polynomial to each pattern segment
    polynomials = []
    for start, end in pattern_boundaries:
        if start < end:
            x = tool_data['Degree'][start:end]
            y = tool_data['Sum of Pixels'][start:end]
        else:
            # For the wrap-around case
            x = np.concatenate([tool_data['Degree'][start:], tool_data['Degree'][:end]])
            y = np.concatenate([tool_data['Sum of Pixels'][start:], tool_data['Sum of Pixels'][:end]])

        # Fit a second-degree polynomial
        coefficients = np.polyfit(x, y, 2)
        polynomial = np.poly1d(coefficients)
        polynomials.append(polynomial)

        # Plot the original data and the fitted polynomial
        plt.figure(figsize=(10, 6))
        plt.plot(tool_data['Degree'], tool_data['Sum of Pixels'], label="Original Sum of Pixels", color='blue')
        plt.plot(x, polynomial(x), label="Fitted Polynomial", color='red')
        plt.title(f"Fitted Polynomial for Pattern {start}-{end}")
        plt.xlabel("Degree")
        plt.ylabel("Sum of Pixels")
        plt.grid(True)
        plt.legend()
        plt.show()

    return polynomials

# Example usage
fit_polynomials_to_patterns(file_path)