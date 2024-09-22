import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Function to perform FFT and plot
def perform_fft(degrees, area, title):
    # Perform FFT
    area_fft = fft(area)
    freqs = fftfreq(len(area), d=(degrees[1] - degrees[0]))
    
    # Compute amplitude (magnitude of FFT)
    amplitude = np.abs(area_fft)
    
    # Find peaks in the amplitude spectrum
    peaks, _ = find_peaks(amplitude, height=np.mean(amplitude)*2)  # Adjust threshold if needed
    
    # Plot the FFT result
    plt.plot(freqs, amplitude, label='Amplitude')
    plt.plot(freqs[peaks], amplitude[peaks], "x", label='Peaks')  # Mark peaks
    plt.title(f'Fourier Transform - {title}')
    plt.xlabel('Frequency (1/Degrees)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Return peak frequencies for further analysis
    return freqs[peaks], amplitude[peaks]

# Load CSV files for broken and intact tools
intact_data = pd.read_csv(r'data\table(intact).csv')  # Replace with your file path
broken_data = pd.read_csv(r'data\table(broken) (1).csv')  # Replace with your file path

# Assuming the CSV files have columns 'Degree' and 'Area'
intact_degrees = intact_data['Degree'].values
intact_area = intact_data['Sum of Pixels'].values
broken_degrees = broken_data['Degree'].values
broken_area = broken_data['Sum of Pixels'].values

# Plot the FFT for both tools
plt.figure(figsize=(12, 6))

# Intact tool
plt.subplot(1, 2, 1)
intact_peaks_freq, intact_peaks_amp = perform_fft(intact_degrees, intact_area, 'Intact Tool')

# Broken tool
plt.subplot(1, 2, 2)
broken_peaks_freq, broken_peaks_amp = perform_fft(broken_degrees, broken_area, 'Broken Tool')

# Show plots
plt.tight_layout()
plt.show()

# Function to convert np.float64 to regular float and format for display
def format_peaks(freqs, amps):
    formatted_peaks = [(float(freq), float(amp)) for freq, amp in zip(freqs, amps)]  # Convert to Python float
    return formatted_peaks

# Clean up the output for display
print("Intact Tool Peaks (Frequency, Amplitude):", format_peaks(intact_peaks_freq, intact_peaks_amp))
print("Broken Tool Peaks (Frequency, Amplitude):", format_peaks(broken_peaks_freq, broken_peaks_amp))


# Simple classification based on number of peaks or amplitude
if len(broken_peaks_freq) > len(intact_peaks_freq) or np.max(broken_peaks_amp) > np.max(intact_peaks_amp) * 1.2:
    print("Tool is likely broken.")
else:
    print("Tool is likely intact.")
