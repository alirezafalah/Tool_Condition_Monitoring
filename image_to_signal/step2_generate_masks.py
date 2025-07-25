import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from .utils.filters import (create_multichannel_mask, fill_holes, 
                          morph_closing, keep_largest_contour)

def run(config):
    """
    Processes pre-blurred images to generate and save the final binary masks
    using the multi-channel segmentation logic.
    """
    input_dir = config['BLURRED_DIR']
    output_dir = config['FINAL_MASKS_DIR']
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No blurred images found in '{input_dir}'. Skipping.")
            return
    except FileNotFoundError:
        print(f"Error: Blurred data directory not found at '{input_dir}'.")
        return

    print(f"Generating final masks for {len(image_files)} images...")
    
    for filename in tqdm(image_files, desc="Generating Masks"):
        image_path = os.path.join(input_dir, filename)
        
        try:
            blurred_image = Image.open(image_path)
            
            # --- Segmentation Pipeline ---
            # 1. Create the initial mask with our new powerful function
            initial_mask = create_multichannel_mask(blurred_image, config)
            if not initial_mask:  # Error handling if mask creation fails
                print(f"Failed to create mask for {filename}.")
                break
            # 2. Apply post-processing to clean up the mask
            filled1 = fill_holes(initial_mask)
            closed = morph_closing(filled1, kernel_size=config['closing_kernel'])
            filled2 = fill_holes(closed)
            final_mask = keep_largest_contour(filled2)
            
            # 3. Save the result
            if final_mask:
                output_path = os.path.join(output_dir, filename)
                final_mask.save(output_path, 'TIFF')

        except Exception as e:
            print(f"Failed to process {filename}. Error: {e}")