### THis code will shift the data by a certain number of degrees to have more datasets to experience

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path =  r"C:\Users\alrfa\OneDrive\Desktop\Ph.D\Paper2_Possible_Methods\data\table(intact).csv"
intact_tool_data = pd.read_csv(file_path)

def circular_shift(data, shift_degrees):
    shifted_data = np.roll(data, shift_degrees)
    return shifted_data

shift_degrees = 30  # Adjust this value to shift the data

shifted_pixels = circular_shift(intact_tool_data['Sum of Pixels'].values, shift_degrees)

plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], intact_tool_data['Sum of Pixels'], label="Original Data", color='blue')
plt.plot(intact_tool_data['Degree'], shifted_pixels, label=f"Shifted Data by {shift_degrees} Degrees", color='orange')
plt.title(f"Original vs Shifted Data by {shift_degrees} Degrees")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()

# If you want to save the shifted data as a new DataFrame:
shifted_df = intact_tool_data.copy()
shifted_df['Shifted Sum of Pixels'] = shifted_pixels

shifted_file_path = f'shifted_data_by_{shift_degrees}_degrees.csv'
shifted_df.to_csv(shifted_file_path, index=False)
