import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def circular_moving_average(file_path, window_size=15, output_file=None):
    """
    Apply a circular moving average to the 'Sum of Pixels' column of a CSV file.
    
    Parameters:
    - file_path (str): Path to the input CSV file (columns: 'Degree', 'Sum of Pixels')
    - window_size (int): Size of the moving average window (default: 15)
    - output_file (str, optional): Path to save the output CSV (if None, no file is saved)
    
    Returns:
    - pd.DataFrame: DataFrame with original data and the moving average column
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract the 'Sum of Pixels' column as a numpy array
    data = df['Sum of Pixels'].values
    
    # Number of data points
    n = len(data)
    
    # Ensure window_size is odd for centered averaging (optional, can remove this check if desired)
    if window_size % 2 == 0:
        print(f"Warning: Window size {window_size} is even. Consider using an odd number for centered averaging.")
    
    # Calculate the half-window size (points before and after the center)
    half_window = (window_size - 1) // 2
    
    # Initialize the output array for moving averages
    moving_avg = np.zeros(n)
    
    # Compute the circular moving average
    for i in range(n):
        # Define the window indices with wrapping
        indices = [(i + j) % n for j in range(-half_window, half_window + 1)]
        # Calculate the average for this window
        moving_avg[i] = np.mean(data[indices])
    
    # Add the moving average to the DataFrame
    df['Moving Average'] = moving_avg
    
    # Optionally save to a new CSV file
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df

# Example usage for a single file
if __name__ == "__main__":
    # Example file path (replace with your actual file)
    input_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-007\table.csv"
    output_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-007\table_processed.csv"
    
    # Apply moving average with window size 15 and assign the result
    result_df = circular_moving_average(input_file, window_size=15, output_file=output_file)
    
    # Plot the original and smoothed data
    plt.figure(figsize=(10, 5))
    plt.plot(result_df['Degree'], result_df['Sum of Pixels'], label='Original Data')  # Use 'Degree' for x-axis
    plt.plot(result_df['Degree'], result_df['Moving Average'], label='Smoothed Data', linestyle='--')
    plt.xlabel('Degree')  # Adjusted to match your data
    plt.ylabel('Sum of Pixels')
    plt.title('Original and Smoothed Data (Window Size = 15)')
    plt.legend()
    plt.grid(True)  # Optional: adds a grid for better readability
    plt.show()