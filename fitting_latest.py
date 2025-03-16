import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import itertools

# Define colors for different segments
colors = ["blue", "green", "purple", "orange", "cyan"]

# Load your data
df_drill = pd.read_csv(r'processed_data\drill_fractured_processed.csv')

# Define the sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * np.radians(x) + C) + D

# Function to calculate R²
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Fallback: Minimize negative R2 if curve_fit fails
def optimize_fallback(x, y, initial_guess):
    def loss(params):
        y_pred = sinusoidal(x, *params)
        return -calculate_r2(y, y_pred)  # Negative R2 to minimize

    bounds = [(0, None), (0, None), (-np.pi, np.pi), (-np.inf, np.inf)]
    result = minimize(loss, initial_guess, bounds=bounds)
    return result.x if result.success else [np.nan, np.nan, np.nan, np.nan]

# Split the data into segments
num_segments = 2
segment_size = len(df_drill) // num_segments
segments = [df_drill.iloc[i * segment_size: (i + 1) * segment_size].reset_index(drop=True)
            for i in range(num_segments)]

# Initialize a list to store results
results = []

# Initialize the plot for all segments
plt.figure(figsize=(12, 8))

# Create a color cycle to handle more segments than predefined colors
color_cycle = itertools.cycle(colors)

for i, segment in enumerate(segments):
    segment_shifted = segment.copy()
    segment_shifted["Degree"] = segment_shifted["Degree"] - segment_shifted["Degree"].iloc[0]
    
    x_data = segment_shifted['Degree'].values
    y_data = segment_shifted['Sum of Pixels'].values

    # Initial guess
    initial_guess = [np.ptp(y_data) / 2, 2 * np.pi / (x_data[-1] - x_data[0]), 0, np.mean(y_data)]

    try:
        # Attempt fitting with curve_fit
        popt, _ = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess, method="trf")
        r2 = calculate_r2(y_data, sinusoidal(x_data, *popt))
    except RuntimeError:
        # Fallback optimization
        popt = optimize_fallback(x_data, y_data, initial_guess)
        r2 = calculate_r2(y_data, sinusoidal(x_data, *popt)) if not np.isnan(popt).any() else 0.0

    results.append({'Segment': i + 1, 'Amplitude (A)': popt[0], 'Frequency (B)': popt[1],
                    'Phase Shift (C)': popt[2], 'Vertical Offset (D)': popt[3], 'R²': r2})

    # Plot the data and fitted curve
    color = next(color_cycle)
    plt.scatter(x_data, y_data, color=color, alpha=0.6)
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit = sinusoidal(x_fit, *popt)
    plt.plot(x_fit, y_fit, color=color, linestyle='--',
             label=f'Segment {i+1} Fit (R²={r2:.3f})')

# Customize the plot
plt.title('Robust Sinusoidal Fit for All Segments')
plt.xlabel('Shifted Degree')
plt.ylabel('Sum of Pixels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame for the results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
