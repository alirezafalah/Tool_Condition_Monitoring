import cv2
import os
import numpy as np


def median_background(bg_dir):
    # List frames from directories
    bg_frames = sorted(os.listdir(bg_dir))

    # Read all background frames into a list
    images = [cv2.imread(os.path.join(bg_dir, frame)) for frame in bg_frames]

    # Stack all images along a new dimension
    stacked_images = np.stack(images, axis=0)

    # Compute the median along the new dimension
    median_img = np.median(stacked_images, axis=0).astype(np.uint8)

    return median_img


def subtract_from_median(tool_dir, median_bg, output_dir, thresh_value, kernel_size):
    tool_frames = sorted(os.listdir(tool_dir))

    # Define the kernel for the closing operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for tool_frame in tool_frames:
        tool_image = cv2.imread(os.path.join(tool_dir, tool_frame))

        # Subtract the median background
        difference = cv2.absdiff(tool_image, median_bg)

        # Convert the difference image to grayscale
        grayscale_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, threshed = cv2.threshold(grayscale_diff, thresh_value, 255, cv2.THRESH_BINARY)

        # Apply the closing operation
        closed_img = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        # Save the closed and thresholded difference
        output_path = os.path.join(output_dir, tool_frame)
        cv2.imwrite(output_path, closed_img)


# Directories
tool_dir = r"C:\Users\alrfa\OneDrive\Desktop\data\2\frames\tool"
bg_dir = r"C:\Users\alrfa\OneDrive\Desktop\data\2\frames\background"
output_dir = r"C:\Users\alrfa\OneDrive\Desktop\data\2\frames\last_try\tresholding\kernel-15"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Compute the median background
median_bg = median_background(bg_dir)

# Subtract tool frames from the median background
# Note: You can adjust the 'thresh_value' and 'kernel_size' arguments to tune the process.
subtract_from_median(tool_dir, median_bg, output_dir, thresh_value=25, kernel_size=15)
