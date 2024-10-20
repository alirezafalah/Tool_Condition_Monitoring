"""
This module performs Savitzky-Golay smoothing, inverts the signal, detects peaks, identifies patterns, 
and returns the segmented data.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

def load_and_smooth_data(file_path, window_length=11, polyorder=2):
    """Load tool data and apply Savitzky-Golay filter to smooth the data."""
    tool_data = pd.read_csv(file_path)
    smoothed_pixels = savgol_filter(tool_data['Sum of Pixels'], window_length=window_length, polyorder=polyorder)
    return tool_data, smoothed_pixels

def detect_peaks_and_minima(smoothed_pixels, prominence=7, distance=45):
    """Detect peaks and identify minima between peaks."""
    significant_peaks, _ = find_peaks(smoothed_pixels, prominence=prominence, distance=distance)
    
    minima = []
    for i in range(len(significant_peaks)):
        if i < len(significant_peaks) - 1:
            segment = smoothed_pixels[significant_peaks[i]:significant_peaks[i + 1]]
        else:
            segment = np.concatenate([smoothed_pixels[significant_peaks[i]:], smoothed_pixels[:significant_peaks[0]]])

        min_index_in_segment = np.argmin(segment)
        if i < len(significant_peaks) - 1:
            minima.append(significant_peaks[i] + min_index_in_segment)
        else:
            minima.append((significant_peaks[i] + min_index_in_segment) % len(smoothed_pixels))

    return significant_peaks, minima

def identify_segments(minima):
    """Identify segments based on minima."""
    pattern_boundaries = []
    for i in range(len(minima)):
        if i < len(minima) - 1:
            start = minima[i]
            end = minima[i + 1]
        else:
            start = minima[i]
            end = minima[0]
        pattern_boundaries.append((start, end))
    return pattern_boundaries

def plot_segments(tool_data, smoothed_pixels, pattern_boundaries):
    """Plot the segmented pattern."""
    plt.figure(figsize=(10,6))
    plt.plot(tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')

    colors = ['green', 'red', 'blue', 'purple', 'yellow']
    for i, (start, end) in enumerate(pattern_boundaries):
        if start < end:
            plt.axvspan(tool_data['Degree'][start], tool_data['Degree'][end], color=colors[i % len(colors)], alpha=0.3)
        else:
            plt.axvspan(tool_data['Degree'][start], tool_data['Degree'].iloc[-1], color=colors[i % len(colors)], alpha=0.3)
            plt.axvspan(tool_data['Degree'][0], tool_data['Degree'][end], color=colors[i % len(colors)], alpha=0.3)

    plt.title("Segmented Pattern")
    plt.xlabel("Degree")
    plt.ylabel("Sum of Pixels")
    plt.grid(True)
    plt.legend()
    plt.show()

def segment_tool_data(file_path):
    """Main function to process and segment tool data."""
    tool_data, smoothed_pixels = load_and_smooth_data(file_path)
    significant_peaks, minima = detect_peaks_and_minima(smoothed_pixels)
    pattern_boundaries = identify_segments(minima)
    plot_segments(tool_data, smoothed_pixels, pattern_boundaries)
    return tool_data, smoothed_pixels, pattern_boundaries
