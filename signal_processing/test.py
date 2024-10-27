import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import spectrogram
import pandas as pd
from utils.smoothing import *

data = pd.read_csv(r"../data/intact.csv")
degree = data['Degree']
pixel_values = data['Sum of Pixels']

# Assuming your smoothed signal is stored in 'smoothed_pixels'
smoothed_pixels = smoothing_gaussian(degree, pixel_values, 1)
smoothed_pixels = np.array(smoothed_pixels)

signal = smoothed_pixels - np.mean(smoothed_pixels)

# Compute the autocorrelation
autocorr = np.correlate(signal, signal, mode='full')
autocorr = autocorr[autocorr.size // 2:]  # Keep the second half only

# Plot the autocorrelation
plt.figure(figsize=(10, 6))
plt.plot(autocorr)
plt.title('Autocorrelation of the Signal')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
