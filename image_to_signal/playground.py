from PIL import Image
from image_utils import display_images
from filters import apply_median_blur, convert_to_hsv, create_hue_mask, create_saturation_mask, create_lab_ab_mask, convert_to_lab
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
    # We use the file path here because apply_median_blur expects it
    blurred_image = apply_median_blur(input_file, kernel_size=13)
    # hsv_image = convert_to_hsv(blurred_image)

    # --- HSV Masking ---
    hsv_image = convert_to_hsv(blurred_image)
    hue_mask_pil = create_hue_mask(hsv_image, HUE_MIN, HUE_MAX)
    saturation_mask_pil = create_saturation_mask(hsv_image, SAT_MIN)
    # Combine the HSV masks
    hsv_mask_final = cv2.bitwise_or(np.array(hue_mask_pil), np.array(saturation_mask_pil))

    # --- LAB Masking ---
    lab_image = convert_to_lab(blurred_image)
    lab_mask_pil = create_lab_ab_mask(lab_image, LAB_AB_MAX)

    # --- Final Combination: (HSV Mask) OR (LAB Mask) ---
    final_mask = cv2.bitwise_or(hsv_mask_final, np.array(lab_mask_pil))
    final_mask_pil = Image.fromarray(final_mask)

    # --- Visualize ---
    original_image = Image.open(input_file)
    display_images([
        (original_image, 'Original'),
        (Image.fromarray(hsv_mask_final), 'HSV Mask'),
        (lab_mask_pil, 'LAB Mask'),
        (final_mask_pil, 'Final Combined Mask')
    ])