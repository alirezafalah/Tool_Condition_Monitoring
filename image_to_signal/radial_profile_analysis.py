import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
INPUT_DIR = 'data/tool014gain10paperBG_final_masks'
# A new folder to save the 500 plots
OUTPUT_PLOT_DIR = 'data/tool014gain10paperBG_radius_profile' 
ROI_HEIGHT = 500 # The number of rows from the tip to analyze
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def main():
    """
    Main function to generate row-wise area profiles for a specified ROI.
    Each generated plot will include a visual guide of the ROI.
    """
    start_time = time.time()
    
    # Ensure the output directory for plots exists
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    try:
        image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"Error: No image files found in '{INPUT_DIR}'.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{INPUT_DIR}'.")
        return

    # --- 1. Extract Row Data from All Images ---
    all_row_data = {}
    
    print(f"Reading {len(image_files)} masks and extracting ROI data...")
    for filename in tqdm(image_files, desc="Processing Masks"):
        try:
            angle = float(filename.split('_')[0])
            mask_np = np.array(Image.open(os.path.join(INPUT_DIR, filename)))
            
            white_pixel_coords = np.where(mask_np == 255)
            if white_pixel_coords[0].size == 0: continue
            
            last_row_idx = white_pixel_coords[0].max()
            first_row_idx = max(0, last_row_idx - ROI_HEIGHT)
            
            roi = mask_np[first_row_idx:last_row_idx, :]
            row_sums = np.sum(roi, axis=1) / 255
            
            if len(row_sums) < ROI_HEIGHT:
                padding = np.zeros(ROI_HEIGHT - len(row_sums))
                row_sums = np.concatenate((padding, row_sums))

            all_row_data[angle] = row_sums

        except (ValueError, IndexError):
            continue

    if not all_row_data:
        print("No data was extracted. Exiting.")
        return

    # --- 2. Convert to DataFrame and Prepare for Plotting ---
    df = pd.DataFrame.from_dict(all_row_data, orient='index')
    df.columns = [f"Row {i - (ROI_HEIGHT - 1)}" for i in range(ROI_HEIGHT)]
    df = df.sort_index()

    # --- 3. Load a Representative Mask for Visuals ---
    # We'll use the mask from the middle of the rotation as our visual example
    representative_mask_path = os.path.join(INPUT_DIR, image_files[len(image_files) // 2])
    rep_mask_np = np.array(Image.open(representative_mask_path))
    rep_coords = np.where(rep_mask_np == 255)
    rep_last_row = rep_coords[0].max()
    rep_first_row = max(0, rep_last_row - ROI_HEIGHT)
    representative_roi = rep_mask_np[rep_first_row:rep_last_row, :]

    # --- 4. Generate and Save a Plot for Each Row ---
    print(f"\nGenerating {ROI_HEIGHT} plots...")
    for i, row_name in enumerate(tqdm(df.columns, desc="Generating Plots")):
        # Create a figure with two subplots: one for the graph, one for the ROI image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [3, 1]})
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # --- Panel 1: The Data Plot ---
        ax1.plot(df.index, df[row_name], marker='.', linestyle='-', markersize=3)
        ax1.set_title(f'Pixel Count vs. Angle for {row_name}', fontsize=16)
        ax1.set_xlabel('Tool Rotation Angle (Degrees)', fontsize=12)
        ax1.set_ylabel('Width (Pixel Count)', fontsize=12)
        ax1.set_xlim(0, 360)
        ax1.grid(True)
        
        # --- Panel 2: The ROI Visual Guide ---
        # Convert the B&W ROI to a color image to draw on
        roi_visual = cv2.cvtColor(representative_roi, cv2.COLOR_GRAY2RGB)
        
        # The index of the row to highlight corresponds to the loop index
        # This is more robust than parsing the name string
        row_to_highlight = i
        if row_to_highlight < roi_visual.shape[0]: # Ensure index is within bounds
            roi_visual[row_to_highlight, :] = [255, 0, 0] # Set the row to red

        ax2.imshow(roi_visual)
        ax2.set_title("ROI Guide", fontsize=16)
        ax2.axis('off') # Hide axes for the image
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{i:03d}_{row_name.replace(' ', '_')}_profile.png"
        output_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)
        plt.savefig(output_path, dpi=150)
        plt.close(fig) # Close the figure to free up memory

    end_time = time.time()
    print(f"\nProcessing complete. Saved {ROI_HEIGHT} plots to '{OUTPUT_PLOT_DIR}'.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()



