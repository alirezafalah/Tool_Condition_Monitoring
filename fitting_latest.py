import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define colors for different segments
colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'red']

# Load your data
df_drill = pd.read_csv(r'processed_data\chamfer_processed.csv')  # Use raw string to handle backslashes

# Define the sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * np.radians(x) + C) + D

# Split the data into segments
num_segments = 4  # Adjust based on your data
segment_size = len(df_drill) // num_segments
segments = [df_drill.iloc[i * segment_size: (i + 1) * segment_size].reset_index(drop=True) for i in range(num_segments)]

# Initialize a list to store results
results = []

# Initialize the plot for all segments
plt.figure(figsize=(12, 8))

for i, segment in enumerate(segments):
    x_data = segment['Degree'].values
    y_data = segment['Sum of Pixels'].values

    # Initial guess for parameters: A, B (frequency), C (phase shift), D (vertical offset)
    initial_guess = [1, 2, 0, 0.5]

    try:
        # Curve fitting
        popt, pcov = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
    except RuntimeError as e:
        print(f"Segment {i + 1}: Curve fitting failed. {e}")
        popt = [np.nan, np.nan, np.nan, np.nan]

    # Store the fitted parameters
    results.append({
        'Segment': i + 1,
        'Amplitude (A)': popt[0],
        'Frequency (B)': popt[1],
        'Phase Shift (C)': popt[2],
        'Vertical Offset (D)': popt[3]
    })

    # Plot the data points
    plt.scatter(x_data, y_data, color=colors[i % len(colors)], label=f'Segment {i + 1} Data', alpha=0.6)

    # Plot the fitted curve if fitting was successful
    if not np.isnan(popt).any():
        x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        y_fit = sinusoidal(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=colors[i % len(colors)], linestyle='--', label=f'Segment {i + 1} Fit')

# Customize the plot
plt.title('Sinusoidal Fit for All Segments')
plt.xlabel('Degree')
plt.ylabel('Sum of Pixels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Display the coefficients table
print("\nSinusoidal Fit Coefficients for Each Segment:")
print(results_df.to_string(index=False))
