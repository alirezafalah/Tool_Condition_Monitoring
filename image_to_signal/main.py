from PIL import Image
from image_utils import display_images
from filters import *
import cv2
import numpy as np

if __name__ == "__main__":
    input_file = 'attached_chip.tiff'
    

    # --- Thresholds ---
    HUE_MIN = 142 # Equal to 200-230 in the 0-360 range
    HUE_MAX = 163 # Equal to 200-230 in the 0-360 range
    SAT_MIN = 38 # Converted from 15%
    LAB_AB_MAX = 126 # From a or b < -2
    
    # --- Pre-processing ---
    blurred_image = apply_median_blur(input_file, kernel_size=13)
    hsv_image = convert_to_hsv(blurred_image)

    hue_mask_pil = create_hue_mask(hsv_image, HUE_MIN, HUE_MAX)
    saturation_mask_pil = create_saturation_mask(hsv_image, SAT_MIN)
    hsv_mask = cv2.bitwise_or(np.array(hue_mask_pil), np.array(saturation_mask_pil))
    hsv_mask_pil = Image.fromarray(hsv_mask)

    # --- Full Post-processing Pipeline ---
    # 1. First Hole Filling
    filled_mask_pil = fill_holes(hsv_mask_pil)
    
    # 2. Morphological Closing
    closed_mask_pil = morph_closing(filled_mask_pil, kernel_size=5)

    # 3. Second Hole Filling
    filled_again_pil = fill_holes(closed_mask_pil)
    
    # 4. Keep Only the Largest Contour
    final_mask_pil = keep_largest_contour(filled_again_pil)

    # --- Visualize ---
    original_image = Image.open(input_file)
    display_images([
        (original_image, 'Original'),
        (hsv_mask_pil, 'Initial HSV Mask'),
        (closed_mask_pil, 'After Closing'),
        (final_mask_pil, 'Final Segmented Mask')
    ])