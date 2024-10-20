import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import Polynomial
from SavitzkY_Pattern_identifier import pattern_boundaries

# File path for the intact tool data
file_path = r"C:\Users\alrfa\OneDrive\Desktop\Ph.D\Paper2_Possible_Methods\data\table(intact).csv"


# Load the intact tool data
intact_tool_data = pd.read_csv(file_path)

# Apply Savitzky-Golay filter to smooth the data
window_length = 11  # Window length for smoothing (must be an odd number)
polyorder = 2       # Degree of polynomial for smoothing
smoothed_pixels = savgol_filter(intact_tool_data['Sum of Pixels'], window_length=window_length, polyorder=polyorder)

# Rescale the entire 'Sum of Pixels' (y-axis) between 0 and 1
y_rescaled = (smoothed_pixels - smoothed_pixels.min()) / (smoothed_pixels.max() - smoothed_pixels.min())

# Select one segment to fit the polynomial (e.g., the first segment)
selected_segment = pattern_boundaries[0]  # Change this to fit another segment if needed
start, end = selected_segment
x_segment = intact_tool_data['Degree'][start:end]
y_segment_rescaled = y_rescaled[start:end]

# Fit a 4th-degree polynomial to the rescaled Sum of Pixels for the selected segment
degree = 4
coefficients = np.polyfit(x_segment, y_segment_rescaled, degree)
polynomial = np.poly1d(coefficients)

# Generate x values for the fitted curve (use the same degree range as the selected segment)
x_fit = np.linspace(x_segment.min(), x_segment.max(), 100)
y_fit = polynomial(x_fit)

# Plot the rescaled smoothed data (no background colors)
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], y_rescaled, label="Rescaled Smoothed Sum of Pixels", color='orange')

# Plot the selected segment as scatter points
plt.scatter(x_segment, y_segment_rescaled, label="Selected Segment (Scatter)", color='blue')

# Plot the fitted polynomial curve
plt.plot(x_fit, y_fit, label=f"Fitted Polynomial (Degree {degree})", color='red')

# Set labels and titles
plt.title(f"Tool Data with Fitted Polynomial on Selected Segment")
plt.xlabel("Degree")
plt.ylabel("Rescaled Sum of Pixels (0 to 1)")
plt.grid(True)
plt.legend()

plt.show()