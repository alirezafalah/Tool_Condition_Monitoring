import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load your data
df_drill = pd.read_csv('path_to_your_data.csv')

# Define the sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * np.radians(x) + C) + D

# Assuming your data has 'Degree' and 'Sum of Pixels' columns
# Split the data into segments
num_segments = 2  # Adjust this based on your data
segment_size = len(df_drill) // num_segments
segments = [df_drill.iloc[i * segment_size: (i + 1) * segment_size] for i in range(num_segments)]

results = []

for i, segment in enumerate(segments):
    x_data = segment['Degree'].values
    y_data = segment['Sum of Pixels'].values

    # Initial guess for parameters: A, B (frequency), C (phase shift), D (vertical offset)
    initial_guess = [1, 2, 0, 0.5]
    
    # Curve fitting
    popt, pcov = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
    results.append({
        'Segment': i + 1,
        'Amplitude (A)': popt[0],
        'Frequency (B)': popt[1],
        'Phase Shift (C)': popt[2],
        'Vertical Offset (D)': popt[3]
    })

    # Plot the fit
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label=f'Segment {i + 1} Data', alpha=0.6)
    plt.plot(x_data, sinusoidal(x_data, *popt), label='Sinusoidal Fit', color='red')
    plt.title(f'Sinusoidal Fit for Segment {i + 1}')
    plt.xlabel('Degree')
    plt.ylabel('Sum of Pixels')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display results
# import ace_tools as tools; tools.display_dataframe_to_user(name="Sinusoidal Fit Results", dataframe=pd.DataFrame(results))