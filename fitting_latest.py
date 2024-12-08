import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import itertools

# Define colors for different segments
colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'red']

# Load your data
df_drill = pd.read_csv(r'processed_data\chamfer_processed.csv')  # Use raw string for Windows paths

# Define the sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * np.radians(x) + C) + D

# Function to calculate R²
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Split the data into segments
num_segments = 4  # Adjust based on your data
segment_size = len(df_drill) // num_segments
segments = [df_drill.iloc[i * segment_size: (i + 1) * segment_size].reset_index(drop=True) for i in range(num_segments)]

# Initialize a list to store results
results = []

# Initialize the plot for all segments
plt.figure(figsize=(12, 8))

# Create a color cycle to handle more segments than predefined colors
color_cycle = itertools.cycle(colors)

for i, segment in enumerate(segments):
    # Shift 'Degree' to start at 0 for each segment
    segment_shifted = segment.copy()
    segment_shifted["Degree"] = segment_shifted["Degree"] - segment_shifted["Degree"].iloc[0]
    
    x_data = segment_shifted['Degree'].values
    y_data = segment_shifted['Sum of Pixels'].values

    # Initial guess for parameters: A, B (frequency), C (phase shift), D (vertical offset)
    initial_guess = [1, 2, 0, 0.5]

    try:
        # Curve fitting
        popt, pcov = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
        
        # Calculate fitted values
        y_fit = sinusoidal(x_data, *popt)
        
        # Calculate R²
        r2 = calculate_r2(y_data, y_fit)
    except RuntimeError as e:
        print(f"Segment {i + 1}: Curve fitting failed. {e}")
        popt = [np.nan, np.nan, np.nan, np.nan]
        r2 = np.nan

    # Store the fitted parameters and R²
    results.append({
        'Segment': i + 1,
        'Amplitude (A)': popt[0],
        'Frequency (B)': popt[1],
        'Phase Shift (C)': popt[2],
        'Vertical Offset (D)': popt[3],
        'R²': r2
    })

    # Plot the data points and fitted curve
    color = next(color_cycle)
    plt.scatter(x_data, y_data, color=color, alpha=0.6)
    
    if not np.isnan(popt).any():
        x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        y_fit_plot = sinusoidal(x_fit, *popt)
        plt.plot(x_fit, y_fit_plot, color=color, linestyle='--')
        
        # Add to legend with R²
        plt.plot([], [], color=color, linestyle='--', label=f'Segment {i + 1} Fit (R²={r2:.3f})')
        plt.plot([], [], color=color, marker='o', linestyle='', label=f'Segment {i + 1} Data')
    else:
        # If fitting failed, only plot data
        plt.plot([], [], color=color, marker='o', linestyle='', label=f'Segment {i + 1} Data (Fit Failed)')

# Customize the plot
plt.title('Sinusoidal Fit for All Segments (Degree Shifted to Start at 0)')
plt.xlabel('Shifted Degree')
plt.ylabel('Sum of Pixels')
plt.grid(True)
plt.tight_layout()

# Create custom legend to avoid duplicate entries
handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicate labels
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')

plt.show()

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Display the coefficients table
print("\nSinusoidal Fit Coefficients for Each Segment:")
print(results_df.to_string(index=False))

# Optionally, save the coefficients table to a CSV file
# results_df.to_csv('sinusoidal_fit_coefficients.csv', index=False)
