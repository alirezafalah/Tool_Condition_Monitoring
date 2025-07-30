from PIL import Image, ImageFilter
import cv2
import numpy as np

# In utils/filters.py

def create_multichannel_mask(image_object, config):
    """
    Creates a sophisticated binary mask based on combined rules from
    both the LAB and HSV color spaces. All thresholds are pulled from the config dict.
    """
    try:
        # --- 1. Prepare Color Spaces ---
        rgb_image_np = np.array(image_object.convert('RGB'))
        lab_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2LAB)
        hsv_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)

        # --- 2. Isolate All 6 Channels ---
        L_channel, a_channel, b_channel = cv2.split(lab_image_np)
        H_channel, S_channel, V_channel = cv2.split(hsv_image_np)

        # --- 3. Create a Boolean Mask for Each Condition from Config ---
        # This now correctly uses the _min and _max keys from your CONFIG
        L_mask = (L_channel >= config['L_threshold_min']) & (L_channel <= config['L_threshold_max'])
        a_mask = (a_channel >= config['a_threshold_min']) & (a_channel <= config['a_threshold_max'])
        b_mask = (b_channel >= config['b_threshold_min']) & (b_channel <= config['b_threshold_max'])
        
        H_mask = (H_channel >= config['h_threshold_min']) & (H_channel <= config['h_threshold_max'])
        S_mask = (S_channel >= config['s_threshold_min']) & (S_channel <= config['s_threshold_max'])
        V_mask = (V_channel >= config['V_threshold_min']) & (V_channel <= config['V_threshold_max'])

        # --- 4. Combine Masks using the Final Logic ---
        # This is the only line you should change here if you find a better *rule*.
        final_boolean_mask = ~L_mask
        
        # --- 5. Convert to Visual Mask and Return ---
        final_mask_visual = final_boolean_mask.astype(np.uint8) * 255
        return Image.fromarray(final_mask_visual)

    except Exception as e:
        print(f"An error occurred during multi-channel masking: {e}")
        return None

# ==============================================================================
# ==================== GENERAL & POST-PROCESSING FILTERS =======================

def apply_median_blur(tiff_path, kernel_size=13):
    # This function remains unchanged
    try:
        with Image.open(tiff_path) as img:
            blurred_img = img.filter(ImageFilter.MedianFilter(size=kernel_size))
            return blurred_img
    except Exception as e:
        print(f"An error occurred during blurring: {e}")
        return None

def fill_holes(binary_mask_object):
    # This function remains unchanged
    try:
        mask = np.array(binary_mask_object.convert('L'))
        mask_floodfill = mask.copy()
        h, w = mask.shape[:2]
        bordered_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_floodfill, bordered_mask, (0, 0), 255);
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        filled_mask = (mask | mask_floodfill_inv)
        return Image.fromarray(filled_mask)
    except Exception as e:
        print(f"An error occurred during hole filling: {e}")
        return None

def morph_closing(binary_mask_object, kernel_size=5):
    # This function remains unchanged
    try:
        mask = np.array(binary_mask_object.convert('L'))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(closing)
    except Exception as e:
        print(f"An error occurred during morphological closing: {e}")
        return None

def morph_opening(binary_mask_object, kernel_size=5):
    """
    Applies a morphological opening operation to a binary mask.
    
    Args:
        binary_mask_object (PIL.Image.Image): A binary mask.
        kernel_size (int): The size of the kernel for the operation.

    Returns:
        PIL.Image.Image: The mask after the opening operation.
    """
    try:
        mask = np.array(binary_mask_object.convert('L'))
        
        # Create the kernel for the morphological operation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply the opening operation
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return Image.fromarray(opening)

    except Exception as e:
        print(f"An error occurred during morphological opening: {e}")
        return None

def keep_largest_contour(binary_mask_object):
    # This function remains unchanged
    try:
        mask = np.array(binary_mask_object.convert('L'))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            new_mask = np.zeros_like(mask)
            cv2.drawContours(new_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
            return Image.fromarray(new_mask)
        else:
            return binary_mask_object
    except Exception as e:
        print(f"An error occurred while finding the largest contour: {e}")
        return None
        return None
    
# ===================================================================
# ==================== BACKGROUND SUBTRACTION =======================

def background_subtraction_absdiff(image_object, background_np, config):
    """
    Performs background subtraction using simple absolute difference on
    grayscale images.
    """
    try:
        sample_gray = cv2.cvtColor(np.array(image_object.convert('RGB')), cv2.COLOR_RGB2GRAY)
        background_gray = cv2.cvtColor(background_np, cv2.COLOR_RGB2GRAY)
        diff_image = cv2.absdiff(sample_gray, background_gray)
        _, binary_mask = cv2.threshold(
            diff_image,
            config['DIFFERENCE_THRESHOLD'],
            255,
            cv2.THRESH_BINARY
        )
        return Image.fromarray(binary_mask)
    except Exception as e:
        print(f"An error occurred during absdiff background subtraction: {e}")
        return None

def background_subtraction_lab(image_object, background_np, config):
    """
    Performs background subtraction in the LAB color space by calculating
    the perceptual color difference (Delta E).
    """
    try:
        sample_lab = cv2.cvtColor(np.array(image_object.convert('RGB')), cv2.COLOR_RGB2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(background_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        delta_E = np.sqrt(np.sum((sample_lab - background_lab)**2, axis=2))
        diff_image = cv2.normalize(delta_E, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, binary_mask = cv2.threshold(
            diff_image,
            config['DIFFERENCE_THRESHOLD'],
            255,
            cv2.THRESH_BINARY
        )
        return Image.fromarray(binary_mask)
    except Exception as e:
        print(f"An error occurred during LAB background subtraction: {e}")
        return None