import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# Make sure filters.py is in the same directory
from filters import (convert_to_hsv, create_hue_mask, create_saturation_mask, 
                     fill_holes, morph_closing, keep_largest_contour)

def main():
    """
    Main function to run the visual testing pipeline.
    It processes pre-blurred images and saves the final masks as image files.
    """
    # --- Configuration ---
    # !! UPDATE THESE PATHS IF NEEDED !!
    # This script reads from the folder of BLURRED images
    INPUT_DIR = 'data/tool014gain10paperBG_blurred'
    # This is where the final masks will be saved
    OUTPUT_DIR = 'data/final_masks'
    
    # --- Processing Parameters ---
    # These should match the parameters in your main analysis script
    PROCESSING_PARAMS = {
        'hue_min': 142,
        'hue_max': 163,
        'sat_min': 38,
        'closing_kernel': 5,
    }

    # --- Main Logic ---
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"Error: No image files found in '{INPUT_DIR}'.")
            print("Please ensure you have run the pre-processing script first.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{INPUT_DIR}'.")
        return

    print(f"Starting visual test: generating final masks for {len(image_files)} images...")
    
    # Loop through each pre-blurred image
    for filename in tqdm(image_files, desc="Generating Masks"):
        image_path = os.path.join(INPUT_DIR, filename)
        
        try:
            # Open the blurred image
            blurred_image = Image.open(image_path)
            
            # --- Apply the rest of the pipeline (everything after blur) ---
            hsv_image = convert_to_hsv(blurred_image)
            
            hue_mask_pil = create_hue_mask(hsv_image, PROCESSING_PARAMS['hue_min'], PROCESSING_PARAMS['hue_max'])
            saturation_mask_pil = create_saturation_mask(hsv_image, PROCESSING_PARAMS['sat_min'])
            
            hsv_mask = cv2.bitwise_or(np.array(hue_mask_pil), np.array(saturation_mask_pil))
            hsv_mask_pil = Image.fromarray(hsv_mask)

            filled_mask_pil = fill_holes(hsv_mask_pil)
            closed_mask_pil = morph_closing(filled_mask_pil, kernel_size=PROCESSING_PARAMS['closing_kernel'])
            filled_again_pil = fill_holes(closed_mask_pil)
            final_mask_pil = keep_largest_contour(filled_again_pil)
            
            # --- Save the Final Mask ---
            if final_mask_pil:
                output_path = os.path.join(OUTPUT_DIR, filename)
                # Save the final mask as a TIFF to the output directory
                final_mask_pil.save(output_path, 'TIFF')

        except Exception as e:
            print(f"Failed to process {filename}. Error: {e}")

    print(f"\nProcessing complete. Final masks saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
