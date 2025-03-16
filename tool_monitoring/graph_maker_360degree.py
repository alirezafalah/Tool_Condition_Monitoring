import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files for broken and intact tools
intact_data = pd.read_csv(r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-007\table.csv")  # Relative path to 'data' folder
# broken_data = pd.read_csv(r'../data/table(broken).csv')  # Relative path to 'data' folder

# Assuming the CSV files have columns 'Degree' and 'Area'
intact_degrees = intact_data['Degree'].values
intact_area = intact_data['Sum of Pixels'].values
# broken_degrees = broken_data['Degree'].values
# broken_area = broken_data['Sum of Pixels'].values

# Define consistent y-axis limits
y_min, y_max = 3500, 8500

# Plot the original patterns (area vs. degrees)
plt.figure(figsize=(12, 6))

# Plot for the intact tool
# plt.subplot(1, 2, 1)
plt.plot(intact_degrees, intact_area, label='Tool', color='blue')
plt.title('Tool Pattern')
plt.xlabel('Degrees')
plt.ylabel('Area')
plt.grid(True)
# plt.ylim(min(intact_area) - 500, max(intact_area) + 500)  # Adjust y-axis limits for zoomed-out view
# plt.xticks(np.arange(0, max(intact_degrees)+10, 10))
plt.ylim(y_min, y_max)

# # Plot for the broken tool
# plt.subplot(1, 2, 2)
# plt.plot(broken_degrees, broken_area, label='Fractured Tool', color='red')
# plt.title('Fractured Tool Pattern')
# plt.xlabel('Degrees')
# plt.ylabel('Area')
# plt.grid(True)
# # plt.ylim(min(broken_area) - 500, max(broken_area) + 500)  # Adjust y-axis limits for zoomed-out view
# plt.ylim(y_min, y_max)

# Show the plots
plt.tight_layout()
plt.show()



