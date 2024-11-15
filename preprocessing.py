import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset for intact tool
df_intact = pd.read_csv(r"data\table(intact).csv")  # Replace with actual path

# Step 1: Scaling for intact tool data
df_intact["Sum of Pixels"] = (df_intact["Sum of Pixels"] - df_intact["Sum of Pixels"].min()) / (
    df_intact["Sum of Pixels"].max() - df_intact["Sum of Pixels"].min()
)

# Save scaled data
df_intact.to_csv("processed_data/scaled_intact.csv", index=False)

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

# Save shifted data
df_intact_shifted.to_csv("processed_data/shifted_intact.csv", index=False)

# Plot shifted data
plt.figure(figsize=(8, 6))
plt.scatter(df_intact_shifted["Degree"], df_intact_shifted["Sum of Pixels"], color="blue", label="Processed data")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.title("Chamfer - minimum shifted to 0 degree")
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Splitting shifted data into adjustable segments
num_segments = 4  # Adjust this value to the desired number of segments
segment_size = len(df_intact_shifted) // num_segments

# Save and plot data divided into segments
plt.figure(figsize=(8, 6))
colors = ["blue", "green", "red", "purple", "orange", "cyan"]  # Add more colors if needed
for i in range(num_segments):
    segment = df_intact_shifted.iloc[i * segment_size: (i + 1) * segment_size]
    segment.to_csv(f"processed_data/segment_{i+1}.csv", index=False)  # Save each segment
    plt.scatter(segment["Degree"], segment["Sum of Pixels"], color=colors[i % len(colors)], label=f"Segment {i+1}")

plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.title(f"Chamfer - Divided by number of segments")
plt.legend()
plt.grid(True)
plt.show()
