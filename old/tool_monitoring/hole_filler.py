import os
import cv2
import numpy as np
import zipfile

def median_background(bg_dir):
    bg_frames = sorted(os.listdir(bg_dir))
    images = [cv2.imread(os.path.join(bg_dir, frame)) for frame in bg_frames]
    stacked_images = np.stack(images, axis=0)
    median_img = np.median(stacked_images, axis=0).astype(np.uint8)
    return median_img

def fill_enclosed_regions(img):
    inverted_img = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(inverted_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape
    filled_img = img.copy()

    for contour in contours:
        at_border = False
        for point in contour:
            if (point[0][0] == 0 or point[0][0] == w-1 or point[0][1] == 0 or point[0][1] == h-1):
                at_border = True
                break
        if not at_border:
            cv2.drawContours(filled_img, [contour], 0, 255, -1)
    return filled_img

def improved_subtraction(tool_dir, median_bg, output_dir, thresh_value=25, kernel_size=5):
    tool_frames = sorted(os.listdir(tool_dir))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for tool_frame in tool_frames:
        tool_image = cv2.imread(os.path.join(tool_dir, tool_frame))
        difference = cv2.absdiff(tool_image, median_bg)
        grayscale_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(grayscale_diff, thresh_value, 255, cv2.THRESH_BINARY)
        closed_img = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        final_img = fill_enclosed_regions(closed_img)
        output_path = os.path.join(output_dir, tool_frame)
        cv2.imwrite(output_path, final_img)

# Directories
background_dir = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E6-001\BG"
tool_dir = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E6-001\TOOL"
output_dir = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E6-001\SUBSTRACT"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Compute the median background
median_bg = median_background(background_dir)

# Subtract tool frames from the median background using the improved method
improved_subtraction(tool_dir, median_bg, output_dir, thresh_value=20, kernel_size=7)
