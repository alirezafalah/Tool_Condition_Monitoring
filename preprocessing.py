import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset for intact tool
df_intact = pd.read_csv(r"data\table(intact).csv")  # Replace with actual path

# Step 1: Scaling for intact tool data
df_intact["Sum of Pixels"] = (df_intact["Sum of Pixels"] - df_intact["Sum of Pixels"].min()) / (
    df_intact["Sum of Pixels"].max() - df_intact["Sum of Pixels"].min()
)

# Plot scaled data for intact tool
plt.figure(figsize=(8, 6))
plt.plot(df_intact["Degree"], df_intact["Sum of Pixels"], marker="o", linestyle="-")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.title("Chamfer - Scaled")
plt.grid(True)
plt.show()

# Step 2: Shifting the minimum of intact data
min_index = df_intact["Sum of Pixels"].idxmin()
df_intact_shifted = pd.concat([df_intact.loc[min_index:], df_intact.loc[:min_index - 1]])
df_intact_shifted = df_intact_shifted.reset_index(drop=True)

# Shift the degree axis to start from 0
df_intact_shifted["Degree"] = df_intact_shifted["Degree"] - df_intact_shifted.iloc[0]["Degree"]
df_intact_shifted.loc[df_intact_shifted["Degree"] < 0, "Degree"] += 360

# Plot shifted data
plt.figure(figsize=(8, 6))
plt.scatter(df_intact_shifted["Degree"], df_intact_shifted["Sum of Pixels"], color="blue", label="Original data")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.title("Chamfer - minimum shifted to 0 degree")
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Splitting shifted data into quadrants
df_intact_quadrant1 = df_intact_shifted.iloc[:90]
df_intact_quadrant2 = df_intact_shifted.iloc[90:180]
df_intact_quadrant3 = df_intact_shifted.iloc[180:270]
df_intact_quadrant4 = df_intact_shifted.iloc[270:]

# Plot data divided into quadrants
plt.figure(figsize=(8, 6))
plt.scatter(df_intact_quadrant1["Degree"], df_intact_quadrant1["Sum of Pixels"], color="blue", label="Quadrant 1")
plt.scatter(df_intact_quadrant2["Degree"], df_intact_quadrant2["Sum of Pixels"], color="green", label="Quadrant 2")
plt.scatter(df_intact_quadrant3["Degree"], df_intact_quadrant3["Sum of Pixels"], color="red", label="Quadrant 3")
plt.scatter(df_intact_quadrant4["Degree"], df_intact_quadrant4["Sum of Pixels"], color="purple", label="Quadrant 4")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.title("Chamfer - Data divided into quadrants")
plt.legend()
plt.grid(True)
plt.show()
