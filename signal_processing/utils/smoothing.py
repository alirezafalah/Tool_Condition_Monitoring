"""
Gaussian Filter for smoothing
"""

import matplotlib.pyplot as plt
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


