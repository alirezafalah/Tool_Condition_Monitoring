from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter to smooth the data
# window_length=21: window size over which the filter will operate
# polyorder=3: degree of the polynomial used to fit the data within the window
smoothed_pixels = savgol_filter(intact_tool_data['Sum of Pixels'], window_length=21, polyorder=3)

# Plot the smoothed data
plt.figure(figsize=(10,6))
plt.plot(intact_tool_data['Degree'], smoothed_pixels, label="Smoothed Sum of Pixels", color='orange')
plt.title("Smoothed Data with Savitzky-Golay Filter")
plt.xlabel("Degree")
plt.ylabel("Sum of Pixels")
plt.grid(True)
plt.legend()
plt.show()
