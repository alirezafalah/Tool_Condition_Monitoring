"""
This code performs Savitzky-Golay smoothing, detects peaks, identifies patterns, and plots both the original and smoothed data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# File path for the intact tool data
file_path = r"C:\Users\alrfa\OneDrive\Desktop\Ph.D\Paper2_Possible_Methods\data\new_tool_pattern_from_broken.csv"

# Load the intact tool data
intact_tool_data = pd.read_csv(file_path)

# Apply Savitzky-Golay filter to smooth the data
# Adjust window_length and polyorder as needed to control the smoothing
window_length = 45  # Window length for smoothing (must be an odd number)
polyorder = 2       # Degree of polynomial for smoothing
smoothed_pixels = savgol_filter(intact_tool_data['Sum of Pixels'], window_length=window_length, polyorder=polyorder)

# Plot both original and smoothed data
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], intact_tool_data['Sum of Pixels'], label="Original Sum of Pixels", color='blue')
plt.plot(intact_tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')
plt.title("Original and Smoothed Intact Tool using Savitzky-Golay Filter")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# Peak detection using find_peaks on the smoothed data
# Adjust prominence, distance, and other parameters to fine-tune peak detection
prominence = 1     # Adjust the prominence to filter significant peaks
distance = 45       # Minimum horizontal distance between peaks
significant_peaks, _ = find_peaks(smoothed_pixels, prominence=prominence, distance=distance)

# Define patterns as peak-to-peak
pattern_boundaries = []

for i in range(len(significant_peaks) - 1):
    start = significant_peaks[i]
    end = significant_peaks[i + 1]
    pattern_boundaries.append((start, end))

# Handle the case where the signal starts before the first peak (from degree 0 to first peak)
if significant_peaks[0] != 0:
    pattern_boundaries.insert(0, (0, significant_peaks[0]))

# Plot the smoothed data with significant peaks and boundaries
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], intact_tool_data['Sum of Pixels'], label="Original Sum of Pixels", color='blue')
plt.plot(intact_tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')

# Plot the significant peak-to-peak boundaries
for (start, end) in pattern_boundaries:
    plt.axvspan(intact_tool_data['Degree'][start], intact_tool_data['Degree'][end], color='green', alpha=0.3)

plt.title("Original and Smoothed Intact Tool with Significant Peak-to-Peak Pattern Boundaries")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# Show the start and end degrees of the significant peak-to-peak patterns
pattern_boundaries_df = pd.DataFrame(pattern_boundaries, columns=['Pattern Start (Degree)', 'Pattern End (Degree)'])
print(pattern_boundaries_df)
