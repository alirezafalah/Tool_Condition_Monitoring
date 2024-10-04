import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load CSV files for broken and intact tools
intact_data = pd.read_csv('data/new_tool_pattern_from_broken.csv')  # Relative path to 'data' folder
broken_data = pd.read_csv('data/table(broken).csv')  # Relative path to 'data' folder

# Assuming the CSV files have columns 'Degree' and 'Sum of Pixels'
intact_degrees = intact_data['Degree'].values.reshape(-1, 1)
intact_area = intact_data['Sum of Pixels'].values
broken_degrees = broken_data['Degree'].values.reshape(-1, 1)
broken_area = broken_data['Sum of Pixels'].values

# Polynomial Regression
degree_of_polynomial = 7  # You can adjust this based on performance

# Polynomial features for intact tool
poly = PolynomialFeatures(degree=degree_of_polynomial)
intact_degrees_poly = poly.fit_transform(intact_degrees)

# Fit polynomial regression model for intact tool
model_intact = LinearRegression()
model_intact.fit(intact_degrees_poly, intact_area)
intact_area_pred = model_intact.predict(intact_degrees_poly)

# Polynomial features for broken tool
broken_degrees_poly = poly.fit_transform(broken_degrees)

# Fit polynomial regression model for broken tool
model_broken = LinearRegression()
model_broken.fit(broken_degrees_poly, broken_area)
broken_area_pred = model_broken.predict(broken_degrees_poly)

# Calculate the Mean Squared Error for intact and broken tools
mse_intact = mean_squared_error(intact_area, intact_area_pred)
mse_broken = mean_squared_error(broken_area, broken_area_pred)

# Print the MSE results
print(f'MSE for Intact Tool: {mse_intact}')
print(f'MSE for Broken Tool: {mse_broken}')

# Plot the original and predicted patterns (area vs. degrees)
plt.figure(figsize=(12, 6))

# Plot for the intact tool
plt.subplot(1, 2, 1)
plt.plot(intact_degrees, intact_area, label='Intact Tool - Actual', color='blue')
plt.plot(intact_degrees, intact_area_pred, label='Intact Tool - Predicted', color='cyan', linestyle='--')
plt.title('Intact Tool Pattern (Polynomial Fit)')
plt.xlabel('Degrees')
plt.ylabel('Area')
plt.grid(True)
plt.legend()

# Plot for the broken tool
plt.subplot(1, 2, 2)
plt.plot(broken_degrees, broken_area, label='Broken Tool - Actual', color='red')
plt.plot(broken_degrees, broken_area_pred, label='Broken Tool - Predicted', color='orange', linestyle='--')
plt.title('Broken Tool Pattern (Polynomial Fit)')
plt.xlabel('Degrees')
plt.ylabel('Area')
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
