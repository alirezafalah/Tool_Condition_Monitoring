from PIL import Image
from image_utils import *
from filters import *
import cv2
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

def process_image(image_path, params):
    """
    Applies the full processing pipeline to a single image and returns the final mask.
    
    Args:
        image_path (str): The path to the input image.
        params (dict): A dictionary of processing parameters.
        
    Returns:
        PIL.Image.Image: The final binary mask of the tool.
    """
    # --- Initial Masking ---
    blurred_image = apply_median_blur(image_path, kernel_size=params['blur_kernel'])
    if blurred_image is None: return None
    
    hsv_image = convert_to_hsv(blurred_image)
    if hsv_image is None: return None
    
    hue_mask_pil = create_hue_mask(hsv_image, params['hue_min'], params['hue_max'])
    saturation_mask_pil = create_saturation_mask(hsv_image, params['sat_min'])
    
    hsv_mask = cv2.bitwise_or(np.array(hue_mask_pil), np.array(saturation_mask_pil))
    hsv_mask_pil = Image.fromarray(hsv_mask)

    # --- Full Post-processing Pipeline ---
    # 1. First Hole Filling
    filled_mask_pil = fill_holes(hsv_mask_pil)
    # 2. Morphological Closing
    closed_mask_pil = morph_closing(filled_mask_pil, kernel_size=params['closing_kernel'])
    # 3. Second Hole Filling
    filled_again_pil = fill_holes(closed_mask_pil)
    # 4. Keep Only the Largest Contour
    final_mask_pil = keep_largest_contour(filled_again_pil)
    
    return final_mask_pil

def main():
    """
    Main function to run the batch processing and generate the CSV file.
    """
    # --- Configuration ---
    INPUT_DIR = 'data/tool014gain10paperBG'  # Directory with your ~400 images
    OUTPUT_CSV = 'data/tool_area_vs_angle.csv'
    
    # --- Processing Parameters ---
    # These are the values we found through testing
    PROCESSING_PARAMS = {
        'blur_kernel': 13,
        'hue_min': 142, # Equal to 200-230 in the 0-360 range
        'hue_max': 163, # Equal to 200-230 in the 0-360 range
        'sat_min': 38, # Converted from 15%
        'lab_ab_max' : 126, # From a or b < -2
        'closing_kernel': 5,
        'images_for_360_deg': 409*(59/60) # The number of images that the tool rotates multiplied by the factor to remove the last 1/60th of a rotation which is extra (we rotate for 366 degrees on each tool)
    }

    # --- Main Logic ---
    # Get a sorted list of image files
    try:
        image_files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(('.tiff', '.tif', '.png', '.jpg'))])
        if not image_files:
            print(f"Error: No image files found in '{INPUT_DIR}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{INPUT_DIR}'. Please create it and add your images.")
        return

    results = []
    angle_step = 360.0 / PROCESSING_PARAMS['images_for_360_deg']
    
    print(f"Starting processing for {len(image_files)} images...")
    
    # Loop through each image with a progress bar
    for i, image_path in enumerate(tqdm(image_files, desc="Processing Images")):
        current_angle = i * angle_step
        
        # Process the image to get the final mask
        final_mask = process_image(image_path, PROCESSING_PARAMS)
        
        if final_mask:
            # Count the white pixels (area)
            mask_np = np.array(final_mask)
            pixel_count = np.sum(mask_np) / 255  # Divide by 255 to count pixels, not sum of intensities
            
            results.append({
                'Angle (Degrees)': current_angle,
                'Area (Pixel Count)': pixel_count
            })

    # --- Save to CSV ---
    if results:
        df = pd.DataFrame(results)
        # Ensure the output directory exists
        output_dir = os.path.dirname(OUTPUT_CSV)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nProcessing complete. Data saved to '{OUTPUT_CSV}'")
    else:
        print("No results were generated. Please check for errors during processing.")

if __name__ == "__main__":
    main()