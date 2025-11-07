import cv2
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

Y_MIN = 2000        # Y-axis minimum for plots
Y_MAX = 6000        # Y-axis maximum for plots

def analyze_tool_images(image_dir, output_csv, offset=50):
    """
    Process images in the given directory to extract a measurement (sum of white pixels)
    from a specified row relative to the bottom-most point of the largest contour.
    The result is saved as a CSV with columns 'Degree' and 'Sum of Pixels'.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Degree', 'Sum of Pixels'])  # CSV header
        
        # Process each image file in sorted order
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    # Extract degree from filename assuming format like toolXXX (e.g., tool001.png)
                    degree = int(filename[4:7])
                except Exception as e:
                    print(f"Filename {filename} is not in the expected format. Skipping.")
                    continue
                
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Could not open image {filename}. Skipping.")
                    continue

                # Find contours in the image (assuming white areas = 255)
                contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    print(f"No contours found in {filename}. Skipping.")
                    continue

                # Find the largest contour and its bottom-most point
                largest_contour = max(contours, key=cv2.contourArea)
                bottom_most_point = max(largest_contour, key=lambda pt: pt[0][1])
                bottom_most_row = bottom_most_point[0][1]
                
                # Define target row with an offset from the bottom-most point
                target_row = max(0, bottom_most_row - offset)
                white_pixel_count = np.sum(image[target_row:, :] == 255)

                writer.writerow([degree, white_pixel_count])
                print(f"Processed {filename}: Degree {degree}, Sum of Pixels {white_pixel_count}")

def plot_original_data(csv_file, graph_file, y_min=Y_MIN, y_max=Y_MAX):
    """
    Load the original CSV data and plot 'Sum of Pixels' vs. 'Degree'. 
    The plot is saved as the provided graph_file.
    """
    df = pd.read_csv(csv_file)
    degrees = df['Degree'].values
    area = df['Sum of Pixels'].values
    
    plt.figure(figsize=(12, 6))
    plt.plot(degrees, area, label='Tool', color='blue')
    plt.title('Tool Pattern')
    plt.xlabel('Degrees')
    plt.ylabel('Area')
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(graph_file)
    print(f"Original data plot saved as {graph_file}")
    plt.close()

def circular_moving_average(file_path, window_size=15, output_file=None):
    """
    Apply a circular moving average on the 'Sum of Pixels' column of the CSV file.
    Saves the processed DataFrame to output_file if provided.
    """
    df = pd.read_csv(file_path)
    data = df['Sum of Pixels'].values
    n = len(data)
    
    if window_size % 2 == 0:
        print(f"Warning: Window size {window_size} is even. Consider using an odd number for centered averaging.")
    half_window = (window_size - 1) // 2
    
    moving_avg = np.zeros(n)
    for i in range(n):
        # Compute indices in a circular manner
        indices = [(i + j) % n for j in range(-half_window, half_window + 1)]
        moving_avg[i] = np.mean(data[indices])
    
    df['Moving Average'] = moving_avg
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Processed CSV saved as {output_file}")
    
    return df

def plot_processed_data(processed_csv, graph_file, window_size=15, y_min=Y_MIN, y_max=Y_MAX):
    """
    Plot the original data and its circular moving average from the processed CSV.
    The plot is saved as the provided graph_file.
    """
    df = pd.read_csv(processed_csv)
    degrees = df['Degree'].values
    original = df['Sum of Pixels'].values
    moving_avg = df['Moving Average'].values
    
    plt.figure(figsize=(10, 5))
    plt.plot(degrees, original, label='Original Data')
    plt.plot(degrees, moving_avg, label='Smoothed Data', linestyle='--')
    plt.xlabel('Degree')
    plt.ylabel('Sum of Pixels')
    plt.title(f'Original and Smoothed Data (Window Size = {window_size})')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(graph_file)
    print(f"Processed data plot saved as {graph_file}")
    plt.close()

if __name__ == "__main__":
    # Set your paths (update these paths as necessary)
    image_directory = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-008\SUBSTRACT"
    output_csv_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-008\table.csv"
    processed_csv_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-008\table_processed.csv"
    graph_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-008\graph.png"
    graph_processed_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-008\graph_processed.png"
    
    offset_pixels = 50  # Offset for image processing
    window_size = 15    # Window size for moving average


    # Step 1: Process images and save the CSV (table.csv)
    analyze_tool_images(image_directory, output_csv_file, offset=offset_pixels)

    # Step 2: Plot original data and save as graph.png
    plot_original_data(output_csv_file, graph_file)

    # Step 3: Apply circular moving average and save processed CSV (table_processed.csv)
    df_processed = circular_moving_average(output_csv_file, window_size=window_size, output_file=processed_csv_file)

    # Step 4: Plot the processed data (original and moving average) and save as graph_processed.png
    plot_processed_data(processed_csv_file, graph_processed_file, window_size=window_size)
