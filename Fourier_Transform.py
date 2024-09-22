import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load CSV files for broken and intact tools
intact_data = pd.read_csv('data/table(intact).csv')  # Intact tool dataset
broken_data = pd.read_csv('data/table(broken) (1).csv')  # Broken tool dataset

intact_degrees = intact_data['Degree'].values
intact_area = intact_data['Sum of Pixels'].values
broken_degrees = broken_data['Degree'].values
broken_area = broken_data['Sum of Pixels'].values

# Perform Fourier Transform for intact tool
intact_area_fft = fft(intact_area)
intact_freqs = fftfreq(len(intact_area), d=(intact_degrees[1] - intact_degrees[0]))

# Perform Fourier Transform for broken tool
broken_area_fft = fft(broken_area)
broken_freqs = fftfreq(len(broken_area), d=(broken_degrees[1] - broken_degrees[0]))

# Plot the results
plt.figure(figsize=(12, 6))

# Plot FFT for Intact Tool
plt.subplot(1, 2, 1)
plt.plot(intact_freqs, np.abs(intact_area_fft))
plt.title('Fourier Transform - Intact Tool')
plt.xlabel('Frequency (1/Degrees)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot FFT for Broken Tool
plt.subplot(1, 2, 2)
plt.plot(broken_freqs, np.abs(broken_area_fft))
plt.title('Fourier Transform - Broken Tool')
plt.xlabel('Frequency (1/Degrees)')
plt.ylabel('Amplitude')
plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()
