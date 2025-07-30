import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from .utils.filters import (create_multichannel_mask, 
                          fill_holes, morph_closing, keep_largest_contour, 
                          background_subtraction_absdiff, background_subtraction_lab)

def run(config):
    """
    Processes pre-blurred images by combining background subtraction and
    multi-channel color masking based on config settings.
    """
    # --- Guardrail: Check if any masking is enabled ---
    method = config.get('BACKGROUND_SUBTRACTION_METHOD', 'none').lower()
    use_mc_mask = config.get('APPLY_MULTICHANNEL_MASK', False)
    if method == 'none' and not use_mc_mask:
        print("Warning: All masking methods are disabled. Exiting.")
        return

    input_dir = config['BLURRED_DIR']
    output_dir = config['FINAL_MASKS_DIR']
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
    except FileNotFoundError:
        print(f"Error: Blurred data directory not found at '{input_dir}'.")
        return

    # --- Load background image ONCE before the loop for efficiency ---
    background_image_np = None
    if method in ['absdiff', 'lab']:
        try:
            bg_path = config['BACKGROUND_IMAGE_PATH']
            background_image_np = np.array(Image.open(bg_path))
            print(f"Using '{method}' method with background image: {bg_path}")
        except FileNotFoundError:
            print(f"Warning: Background image not found for '{method}'. Disabling subtraction.")
            method = 'none'

    print(f"Generating final masks for {len(image_files)} images...")
    
    for filename in tqdm(image_files, desc="Generating Masks"):
        image_path = os.path.join(input_dir, filename)
        
        try:
            blurred_image = Image.open(image_path)
            
            # --- Segmentation Pipeline ---
            bg_mask_np = None
            color_mask_np = None
            # 1. Perform background subtraction with the selected method
            if method == 'absdiff' and background_image_np is not None:
                bg_mask_pil = background_subtraction_absdiff(blurred_image, background_image_np, config)
                if bg_mask_pil: bg_mask_np = np.array(bg_mask_pil)
            elif method == 'lab' and background_image_np is not None:
                bg_mask_pil = background_subtraction_lab(blurred_image, background_image_np, config)
                if bg_mask_pil: bg_mask_np = np.array(bg_mask_pil)

            # 2. Perform multi-channel color masking (if enabled)
            if use_mc_mask:
                color_mask_pil = create_multichannel_mask(blurred_image, config)
                if color_mask_pil:
                    color_mask_np = np.array(color_mask_pil)

            # 3. Combine the masks based on what's enabled
            if bg_mask_np is not None and color_mask_np is not None:
                initial_mask_np = cv2.bitwise_or(bg_mask_np, color_mask_np)
            elif bg_mask_np is not None:
                initial_mask_np = bg_mask_np
            elif color_mask_np is not None:
                initial_mask_np = color_mask_np
            else:
                print(f"Could not generate any mask for {filename}. Skipping.")
                continue
            
            initial_mask = Image.fromarray(initial_mask_np)

            # 4. Apply post-processing to clean up the mask
            filled1 = fill_holes(initial_mask)
            closed = morph_closing(filled1, kernel_size=config['closing_kernel'])
            largest_contour = keep_largest_contour(closed)
            filled2 = fill_holes(largest_contour)
            final_mask = filled2
            
            # 5. Save the result
            if final_mask:
                output_path = os.path.join(output_dir, filename)
                final_mask.save(output_path, 'TIFF')

        except Exception as e:
            print(f"Failed to process {filename}. Error: {e}")