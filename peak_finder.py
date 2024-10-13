"""
This code will find peaks and tries to identify the number of patterns in the raw data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = r"C:\Users\alrfa\OneDrive\Desktop\Ph.D\Paper2_Possible_Methods\data\table(intact).csv"

intact_tool_data = pd.read_csv(file_path)

# Peak detection using find_peaks on the raw data (no smoothing applied)
prominence = 60     # Adjust the prominence to filter significant peaks
distance = 60      # Minimum horizontal distance between peaks
significant_peaks, _ = find_peaks(intact_tool_data['Sum of Pixels'], prominence=prominence, distance=distance)

# Define patterns as peak-to-peak
pattern_boundaries = []

for i in range(len(significant_peaks) - 1):
    start = significant_peaks[i]
    end = significant_peaks[i + 1]
    pattern_boundaries.append((start, end))

# Handle the case where the signal starts before the first peak (from degree 0 to first peak)
if significant_peaks[0] != 0:
    pattern_boundaries.insert(0, (0, significant_peaks[0]))

# Plot the raw data with significant peaks and boundaries
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], intact_tool_data['Sum of Pixels'], label="Sum of Pixels", color='blue')

# Plot the significant peak-to-peak boundaries
for (start, end) in pattern_boundaries:
    plt.axvspan(intact_tool_data['Degree'][start], intact_tool_data['Degree'][end], color='green', alpha=0.3)

plt.title("Raw Data with Significant Peak-to-Peak Pattern Boundaries")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# Show the start and end degrees of the significant peak-to-peak patterns
pattern_boundaries_df = pd.DataFrame(pattern_boundaries, columns=['Pattern Start (Degree)', 'Pattern End (Degree)'])
print(pattern_boundaries_df)
