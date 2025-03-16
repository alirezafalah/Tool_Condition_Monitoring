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
                degree = int(filename[4:7])  # Adjusted to toolXXX (e.g., tool001); modify if needed
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Could not open image {filename}. Skipping.")
                    continue

                # Find contours in the image (white areas are 255)
                contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    print(f"No contours found in {filename}. Skipping.")
                    continue
                
                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get the bottom-most point of the largest contour
                bottom_most_point = max(largest_contour, key=lambda pt: pt[0][1])  # pt[0][1] is the y-coordinate
                bottom_most_row = bottom_most_point[0][1]  # Extract y-coordinate
                
                # Define the target row based on the offset
                target_row = max(0, bottom_most_row - offset)

                # Count white pixels from target_row to the bottom of the image
                white_pixel_count = np.sum(image[target_row:, :] == 255)

                # Write the result to the CSV
                writer.writerow([degree, white_pixel_count])
                print(f"Processed {filename}: Degree {degree}, Sum of Pixels {white_pixel_count}")

# User parameters
image_directory = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-007\SUBSTRACT"  # Update this path
output_csv_file = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E4-007\table.csv"  # Update this path
offset_pixels = 50  # Adjustable offset for row above the bottom-most white pixel

# Run analysis
analyze_tool_images(image_directory, output_csv_file, offset=offset_pixels)