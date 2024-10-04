import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Load CSV files for broken and intact tools
intact_data = pd.read_csv('data/new_tool_pattern_from_broken.csv')  # Relative path to 'data' folder
broken_data = pd.read_csv('data/table(broken).csv')  # Relative path to 'data' folder

# Assuming the CSV files have columns 'Degree' and 'Sum of Pixels'
intact_degrees = intact_data['Degree'].values
intact_area = intact_data['Sum of Pixels'].values
broken_degrees = broken_data['Degree'].values
broken_area = broken_data['Sum of Pixels'].values

# Fit a spline to the intact tool data
spline_degree = 5  # Degree of the spline (cubic, quartic, etc.)
spl_intact = UnivariateSpline(intact_degrees, intact_area, k=spline_degree)

# Use the spline of the intact tool to make predictions on the broken tool's degrees
intact_area_pred_broken = spl_intact(broken_degrees)  # Predict intact behavior at the broken tool's degrees

# Calculate the difference between actual broken tool area and predicted intact area
residuals = broken_area - intact_area_pred_broken

# Plot the spline fit and the residuals to see deviations

plt.figure(figsize=(12, 8))

# Plot for the broken tool with intact spline fit
plt.subplot(2, 1, 1)
plt.plot(broken_degrees, broken_area, label='Broken Tool - Actual', color='red')
plt.plot(broken_degrees, intact_area_pred_broken, label='Predicted Intact Tool (on Broken)', color='orange', linestyle='--')
plt.title('Broken Tool vs Predicted Intact Tool (Spline Fit)')
plt.xlabel('Degrees')
plt.ylabel('Area')
plt.grid(True)
plt.legend()

# Plot the residuals (difference between broken tool and predicted intact tool)
plt.subplot(2, 1, 2)
plt.plot(broken_degrees, residuals, label='Residuals (Broken - Intact)', color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals (Difference Between Broken Tool and Predicted Intact Tool)')
plt.xlabel('Degrees')
plt.ylabel('Difference in Area')
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Optionally, you can analyze the residuals for large deviations
# Define a threshold for identifying breakage (e.g., 100 pixels difference)
threshold = 100
significant_deviations = np.abs(residuals) > threshold

# Print the degrees where the residuals exceed the threshold (indicating breakage)
print(f'Degrees with significant deviations (above threshold {threshold}):')
print(broken_degrees[significant_deviations])
