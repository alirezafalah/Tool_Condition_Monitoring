### This code will shift the data by a certain number of degrees and overwrite the original 'Sum of Pixels' values, but still show the original data in the plot

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r"C:\Users\alrfa\OneDrive\Desktop\Ph.D\Paper2_Possible_Methods\data\broken.csv"
intact_tool_data = pd.read_csv(file_path)

def circular_shift(data, shift_degrees):
    shifted_data = np.roll(data, shift_degrees)
    return shifted_data

shift_degrees = 20  # Adjust this value to shift the data

# Store original values for plotting
original_pixels = intact_tool_data['Sum of Pixels'].values.copy()

# Shift the 'Sum of Pixels' data and overwrite the original values
intact_tool_data['Sum of Pixels'] = circular_shift(intact_tool_data['Sum of Pixels'].values, shift_degrees)

# Plot both the original and shifted data
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], original_pixels, label="Original Data", color='blue')
plt.plot(intact_tool_data['Degree'], intact_tool_data['Sum of Pixels'], label=f"Shifted Data by {shift_degrees} Degrees", color='orange')
plt.title(f"Original vs Shifted Data by {shift_degrees} Degrees")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# Save the modified DataFrame with shifted values to a CSV file
shifted_file_path = f'shifted_data_by_{shift_degrees}_degrees.csv'
intact_tool_data.to_csv(shifted_file_path, index=False)
