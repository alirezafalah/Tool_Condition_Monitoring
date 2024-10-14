"""
This code performs Savitzky-Golay smoothing, inverts the signal, detects peaks, identifies patterns, and plots both the original and smoothed data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

# File path for the intact tool data
file_path = r"C:\Users\alrfa\OneDrive\Desktop\Ph.D\Paper2_Possible_Methods\shifted_data_by_50_degrees.csv"

# Load the intact tool data
intact_tool_data = pd.read_csv(file_path)

# Apply Savitzky-Golay filter to smooth the data
# Adjust window_length and polyorder as needed to control the smoothing
window_length = 11  # Window length for smoothing (must be an odd number)
polyorder = 2       # Degree of polynomial for smoothing
smoothed_pixels = savgol_filter(intact_tool_data['Sum of Pixels'], window_length=window_length, polyorder=polyorder)

# Invert the smoothed signal
inverted_smoothed_pixels = -smoothed_pixels

# Plot both original, smoothed, and inverted-smoothed data
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], intact_tool_data['Sum of Pixels'], label="Original Sum of Pixels", color='blue')
plt.plot(intact_tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')
# plt.plot(intact_tool_data['Degree'], inverted_smoothed_pixels, label="Inverted Smoothed Sum of Pixels", color='green')
plt.title("Original and Smoothed, Intact Tool Data")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# Peak detection using find_peaks on the inverted smoothed data
# Adjust prominence, distance, and other parameters to fine-tune peak detection
prominence = 7     # Adjust the prominence to filter significant peaks
distance = 45      # Minimum horizontal distance between peaks
significant_peaks, _ = find_peaks(smoothed_pixels, prominence=prominence, distance=distance)

# Define patterns as breaking-point to breaking-point
minima = []

for i in range(len(significant_peaks)):
    # If not the last peak, find minima between successive peaks
    if i < len(significant_peaks) - 1:
        segment = smoothed_pixels[significant_peaks[i]:significant_peaks[i + 1]]
    else:
        # For the wrap-around case (last peak to first peak)
        segment = np.concatenate([smoothed_pixels[significant_peaks[i]:], smoothed_pixels[:significant_peaks[0]]])

    # Find the minimum value's index in the current segment
    min_index_in_segment = np.argmin(segment)
    
    # Adjust index based on the segment being part of the overall signal
    if i < len(significant_peaks) - 1:
        minima.append(significant_peaks[i] + min_index_in_segment)
    else:
        minima.append((significant_peaks[i] + min_index_in_segment) % len(smoothed_pixels))

# Define patterns based on the minima, with wrap-around case
pattern_boundaries = []
for i in range(len(minima)):
    if i < len(minima) - 1:
        start = minima[i]
        end = minima[i + 1]
    else:
        # Last segment wraps around to the first minimum
        start = minima[i]
        end = minima[0]
    pattern_boundaries.append((start, end))

# Plot the inverted smoothed data with significant peaks and boundaries
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')

colors = ['green', 'red', 'blue', 'purple', 'yellow']

for i, (start, end) in enumerate(pattern_boundaries):
    if start < end:
        plt.axvspan(intact_tool_data['Degree'][start], intact_tool_data['Degree'][end], color=colors[i % len(colors)], alpha=0.3)
    else:
        # For the wrap-around case
        plt.axvspan(intact_tool_data['Degree'][start], intact_tool_data['Degree'].iloc[-1], color=colors[i % len(colors)], alpha=0.3)
        plt.axvspan(intact_tool_data['Degree'][0], intact_tool_data['Degree'][end], color=colors[i % len(colors)], alpha=0.3)

plt.title("Segmented the pattern")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# Show the start and end degrees of the significant peak-to-peak patterns
pattern_boundaries_df = pd.DataFrame(pattern_boundaries, columns=['Pattern Start (Degree)', 'Pattern End (Degree)'])
print(pattern_boundaries_df)
