import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

def smoothing_gaussian(degree, pixel_values, sigma):
    smoothed_pixels = gaussian_filter1d(pixel_values, sigma)

    # Plot the original and smoothed data for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(degree, pixel_values, label='Original Data', linestyle='--')
    plt.plot(degree, smoothed_pixels, label='Smoothed Data', color='red')
    plt.xlabel('Degree')
    plt.ylabel('Sum of Pixels')
    plt.title('Gaussian Smoothing of Pixel Data')
    plt.legend()
    plt.show()

    return smoothed_pixels

def smoothing_moving_average(degree, pixel_value, window_size):

    # Apply the moving window (boxcar) filter by calculating the rolling mean
    smoothed_pixels_boxcar = pixel_value.rolling(window=window_size, center=True).mean()

    # Plot the original and smoothed data for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(degree, pixel_value, label='Original Data', linestyle='--')
    plt.plot(degree, smoothed_pixels_boxcar, label='Smoothed Data (Boxcar)', color='red')
    plt.xlabel('Degree')
    plt.ylabel('Sum of Pixels')
    plt.title('Moving Average Smoothing of Pixel Data')
    plt.legend()
    plt.show()

    return smoothed_pixels_boxcar


from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def savgol_peak_finder(data_path):
    """
    This code performs Savitzky-Golay smoothing with wrapping, detects peaks, detects breaking points between the peaks, identifies patterns.
    """
    tool_data = pd.read_csv(data_path)
    pixels = tool_data['Sum of Pixels'].values
    
    # Parameters for Savitzky-Golay filter
    window_length = 31  # Window length for smoothing (must be an odd number)
    polyorder = 2       # Degree of polynomial for smoothing
    
    # Wrap the data by extending on both sides
    extended_pixels = np.concatenate((pixels[-window_length//2:], pixels, pixels[:window_length//2]))
    
    # Apply Savitzky-Golay filter to the extended data
    smoothed_extended_pixels = savgol_filter(extended_pixels, window_length=window_length, polyorder=polyorder)
    
    # Trim the smoothed data to match the original length
    smoothed_pixels = smoothed_extended_pixels[window_length//2: -window_length//2]
    
    # Plot both original and wrapped smoothed data
    plt.figure(figsize=(10,6))
    plt.plot(tool_data['Degree'], tool_data['Sum of Pixels'], label="Original Sum of Pixels", color='blue')
    plt.plot(tool_data['Degree'], smoothed_pixels, label="Wrapped Smoothed Sum of Pixels", color='orange')
    plt.title("Original and Wrapped Smoothed, Intact Tool Data")
    plt.xlabel("Degree")
    plt.ylabel("Sum of Pixels")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage:
# savgol_peak_finder('data/tool_data.csv')



    ### PEAK FINDER:

    # # Find the largest vertical difference between consecutive points to set the prominence
    # differences = np.abs(np.diff(smoothed_pixels))
    # max_difference = np.max(differences)

    # # Set prominence greater than the largest difference to avoid detecting local peaks
    # prominence = max_difference + 1  # Ensure prominence is larger than the largest difference
    # distance = 45                    # Minimum horizontal distance between peaks
    # significant_peaks, _ = find_peaks(smoothed_pixels, prominence=prominence, distance=distance)

    # peak_degrees = tool_data['Degree'].iloc[significant_peaks].values
    # peak_values = smoothed_pixels[significant_peaks]

    # # Create a DataFrame to display the peak degrees and values
    # peaks_df = pd.DataFrame({
    #     'Peak Degree': peak_degrees,
    #     'Peak Value (Sum of Pixels)': peak_values
    # })

    # print("Peaks detected:")
    # print(peaks_df)

    # # Define patterns as breaking-point to breaking-point
    # minima = []

    # for i in range(len(significant_peaks)):
    #     # If not the last peak, find minima between successive peaks
    #     if i < len(significant_peaks) - 1:
    #         segment = smoothed_pixels[significant_peaks[i]:significant_peaks[i + 1]]
    #     else:
    #         # For the wrap-around case (last peak to first peak)
    #         segment = np.concatenate([smoothed_pixels[significant_peaks[i]:], smoothed_pixels[:significant_peaks[0]]])

    #     # Find the minimum value's index in the current segment
    #     min_index_in_segment = np.argmin(segment)
        
    #     # Adjust index based on the segment being part of the overall signal
    #     if i < len(significant_peaks) - 1:
    #         minima.append(significant_peaks[i] + min_index_in_segment)
    #     else:
    #         minima.append((significant_peaks[i] + min_index_in_segment) % len(smoothed_pixels))

    # # Define patterns based on the minima, with wrap-around case
    # pattern_boundaries = []
    # for i in range(len(minima)):
    #     if i < len(minima) - 1:
    #         start = minima[i]
    #         end = minima[i + 1]
    #     else:
    #         # Last segment wraps around to the first minimum
    #         start = minima[i]
    #         end = minima[0]
    #     pattern_boundaries.append((start, end))

    # # Plot the smoothed data with significant peaks and boundaries
    # plt.figure(figsize=(10,6))
    # plt.plot(tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')

    # colors = ['green', 'red', 'blue', 'purple', 'yellow']

    # for i, (start, end) in enumerate(pattern_boundaries):
    #     if start < end:
    #         plt.axvspan(tool_data['Degree'][start], tool_data['Degree'][end], color=colors[i % len(colors)], alpha=0.3)
    #     else:
    #         # For the wrap-around case
    #         plt.axvspan(tool_data['Degree'][start], tool_data['Degree'].iloc[-1], color=colors[i % len(colors)], alpha=0.3)
    #         plt.axvspan(tool_data['Degree'][0], tool_data['Degree'][end], color=colors[i % len(colors)], alpha=0.3)

    # plt.title("Segmented the pattern")
    # plt.xlabel("Degree")
    # plt.ylabel("Sum of Pixels")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Show the start and end degrees of the significant peak-to-peak patterns
    # pattern_boundaries_df = pd.DataFrame(pattern_boundaries, columns=['Pattern Start (Degree)', 'Pattern End (Degree)'])
    # print(pattern_boundaries_df)

    # return smoothed_pixels, pattern_boundaries
