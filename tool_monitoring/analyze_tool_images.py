import cv2
import numpy as np
import os
import csv

def analyze_tool_images(image_dir, output_csv, offset=50):
    # Prepare the output CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Degree', 'Sum of Pixels'])  # Header
        
        # Iterate over sorted images in the folder
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Extract degree from filename (assuming format toolXXX where XXX is the degree)
                degree = int(filename[4:7])  # Assumes naming format toolXXX (e.g., tool001)
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Could not open image {filename}. Skipping.")
                    continue

                # Find bottom-most white pixel in the image
                white_pixels = np.where(image == 255)
                if len(white_pixels[0]) == 0:
                    print(f"No white pixels found in {filename}. Skipping.")
                    continue
                
                bottom_most_white_pixel_row = np.max(white_pixels[0])
                target_row = max(0, bottom_most_white_pixel_row - offset)

                # Count white pixels from target_row to the bottom of the image
                white_pixel_count = np.sum(image[target_row:, :] == 255)

                # Write the result to the CSV
                writer.writerow([degree, white_pixel_count])
                print(f"Processed {filename}: Degree {degree}, Sum of Pixels {white_pixel_count}")

# User parameters
image_directory = r"data/subtract(intact)"  # Update this path to where the images are stored
output_csv_file = r"data/drill_intact.csv"  # Update this path for the output CSV
offset_pixels = 50  # Adjustable offset for row above the bottom-most white pixel

# Run analysis
analyze_tool_images(image_directory, output_csv_file, offset=offset_pixels)
