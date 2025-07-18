import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION PARAMETERS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Directories
INPUT_DIR = 'data/final_masks'  # Directory with your final mask images
OUTPUT_CSV = 'data/roi_area_vs_angle.csv'
OUTPUT_PLOT = 'data/roi_area_vs_angle_plot.png'

# ROI Definition
ROI_HEIGHT = 300  # Number of rows up from the last white pixel to define the ROI

# Data Analysis
IMAGES_FOR_360_DEG = 409 * (59 / 60) # Number of images for a full rotation
OUTLIER_STD_DEV_FACTOR = 2.0 # Number of standard deviations to define an outlier

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def rename_masks_by_angle(input_dir):
    """
    Renames mask files in a directory to reflect their rotation angle.
    e.g., '000.98_degrees.tiff'. This function is designed to be run only once.
    """
    print("Checking if mask files need renaming...")
    try:
        # Get a list of files, ignoring potential subdirectories
        image_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print("No mask files found to rename.")
            return

        # Check if files appear to be already renamed
        if '_degrees.tiff' in image_files[0]:
            print("Files appear to be already renamed. Skipping.")
            return

    except FileNotFoundError:
        print(f"Error: Input directory for renaming not found at '{input_dir}'.")
        return

    angle_step = 360.0 / IMAGES_FOR_360_DEG
    print("Renaming files to include their rotation angle...")

    for i, filename in enumerate(tqdm(image_files, desc="Renaming Masks")):
        current_angle = i * angle_step
        # Format with leading zeros for correct alphabetical sorting (e.g., 001.23)
        new_filename = f"{current_angle:07.2f}_degrees.tiff"
        
        old_path = os.path.join(input_dir, filename)
        new_path = os.path.join(input_dir, new_filename)
        
        try:
            os.rename(old_path, new_path)
        except OSError as e:
            print(f"\nError renaming {filename}: {e}")
            continue
    
    print("File renaming complete.")


def find_roi_and_calculate_area(mask_np):
    """
    Finds the Region of Interest (ROI) and calculates the white pixel area within it.
    """
    # Find the y-coordinates of all white pixels
    white_pixel_coords = np.where(mask_np == 255)
    
    if white_pixel_coords[0].size == 0:
        return 0 # No white pixels found

    # Find the last row (max y-coordinate) with a white pixel
    last_row = white_pixel_coords[0].max()
    
    # Define the top of the ROI
    first_row = max(0, last_row - ROI_HEIGHT)
    
    # Create the ROI by slicing the numpy array
    roi = mask_np[first_row:last_row, :]
    
    # Calculate the area within the ROI
    roi_area = np.sum(roi) / 255
    
    return roi_area

def analyze_masks_in_roi(input_dir):
    """
    Processes all masks in a directory to calculate the area within the defined ROI.
    """
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"Error: No mask files found in '{input_dir}'.")
            return None
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'.")
        return None

    results = []
    angle_step = 360.0 / IMAGES_FOR_360_DEG
    
    print(f"Starting ROI analysis for {len(image_files)} masks...")
    
    for i, filename in enumerate(tqdm(image_files, desc="Analyzing ROI")):
        image_path = os.path.join(input_dir, filename)
        mask_image = Image.open(image_path)
        mask_np = np.array(mask_image)
        
        roi_area = find_roi_and_calculate_area(mask_np)
        current_angle = i * angle_step
        
        results.append({
            'Angle (Degrees)': current_angle,
            'ROI Area (Pixels)': roi_area
        })
        
    return pd.DataFrame(results)

def filter_and_plot_data(df):
    """
    Filters outliers from the DataFrame and generates a plot.
    """
    if df is None or df.empty:
        print("Cannot plot. No data was generated.")
        return
        
    # --- Filter Outliers ---
    area_mean = df['ROI Area (Pixels)'].mean()
    area_std = df['ROI Area (Pixels)'].std()
    
    lower_bound = area_mean - (OUTLIER_STD_DEV_FACTOR * area_std)
    upper_bound = area_mean + (OUTLIER_STD_DEV_FACTOR * area_std)
    
    inliers = df[(df['ROI Area (Pixels)'] >= lower_bound) & (df['ROI Area (Pixels)'] <= upper_bound)]
    outliers_count = len(df) - len(inliers)
    
    print(f"Removed {outliers_count} outliers based on {OUTLIER_STD_DEV_FACTOR}x standard deviation.")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(inliers['Angle (Degrees)'], inliers['ROI Area (Pixels)'], 
            marker='.', linestyle='-', markersize=4, label='Filtered ROI Area')

    # --- Formatting ---
    ax.set_title('Tool ROI Area vs. Rotation Angle', fontsize=18, fontweight='bold')
    ax.set_xlabel('Angle (Degrees)', fontsize=14)
    ax.set_ylabel(f'Projected Area in ROI (Pixel Count)', fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 360) 
    plt.tight_layout()

    # --- Save Plot ---
    output_dir = os.path.dirname(OUTPUT_PLOT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to '{OUTPUT_PLOT}'")
    plt.show()

def main():
    """
    Main execution function.
    """
    # 1. Rename files to include their angle (this will only run once)
    rename_masks_by_angle(INPUT_DIR)

    # 2. Analyze masks to get ROI area data
    df_roi_data = analyze_masks_in_roi(INPUT_DIR)
    
    if df_roi_data is not None:
        # 3. Save the raw ROI data to CSV
        output_dir = os.path.dirname(OUTPUT_CSV)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df_roi_data.to_csv(OUTPUT_CSV, index=False)
        print(f"ROI data saved to '{OUTPUT_CSV}'")
        
        # 4. Filter and plot the data
        filter_and_plot_data(df_roi_data)

if __name__ == "__main__":
    main()
