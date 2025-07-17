from PIL import Image, ImageFilter
import cv2
import numpy as np

def apply_median_blur(tiff_path, kernel_size=13):
    """
    Applies a median blur to a TIFF image and returns the result as an object.
    
    Args:
        tiff_path (str): Path to the input TIFF file.
        kernel_size (int): The size of the median filter kernel (must be an odd integer).
    
    Returns:
        PIL.Image.Image: The blurred image object, or None if an error occurs.
    """
    try:
        with Image.open(tiff_path) as img:
            # Apply the MedianFilter and return the new image object
            blurred_img = img.filter(ImageFilter.MedianFilter(size=kernel_size))
            print(f"Successfully applied median blur with kernel size {kernel_size}.")
            return blurred_img
        return None
    except FileNotFoundError:
        print(f"Error: Input file not found at '{tiff_path}'")
        return None
    except Exception as e:
        print(f"An error occurred during blurring: {e}")
        return None
    
def convert_to_hsv(image_object):
    """Converts a PIL Image object to the HSV color space."""
    try:
        return image_object.convert('HSV')
    except Exception as e:
        print(f"An error occurred during HSV conversion: {e}")
        return None
    
def convert_to_lab(image_object):
    """Converts a PIL Image object to the LAB color space."""
    try:
        return image_object.convert('LAB')
    except Exception as e:
        print(f"An error occurred during LAB conversion: {e}")
        return None

def fill_holes(binary_mask_object):
    """
    Fills holes in a binary mask using the flood fill algorithm.
    
    Args:
        binary_mask_object (PIL.Image.Image): A binary (black and white) mask.

    Returns:
        PIL.Image.Image: The mask with holes filled.
    """
    try:
        # Convert PIL image to OpenCV format
        mask = np.array(binary_mask_object.convert('L'))
        
        # Create a copy for flood filling
        mask_floodfill = mask.copy()
        
        # We need a border for floodFill to work correctly
        h, w = mask.shape[:2]
        # Create a new mask with a 2-pixel border
        bordered_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from the top-left corner (an external point)
        cv2.floodFill(mask_floodfill, bordered_mask, (0, 0), 255);
        
        # Invert the flood-filled image
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        
        # Combine the original mask with the inverted flood-filled image
        # This fills the holes
        filled_mask = (mask | mask_floodfill_inv)
        
        return Image.fromarray(filled_mask)

    except Exception as e:
        print(f"An error occurred during hole filling: {e}")
        return None
    

# ==============================================================================
# ==================== HSV MASKS - COLOR RANGE SEGMENTATION ====================

def create_hue_mask(hsv_image_object, h_min, h_max):
    """Creates a mask based on a specific Hue range."""
    try:
        # OpenCV works with numpy arrays
        hsv_cv_image = np.array(hsv_image_object)
        
        # In Pillow/OpenCV, Hue is 0-255, not 0-360
        # Your values (200-230) are already in this range.
        lower_bound = np.array([h_min, 0, 0])
        upper_bound = np.array([h_max, 255, 255])
        
        mask = cv2.inRange(hsv_cv_image, lower_bound, upper_bound)
        return Image.fromarray(mask) # Return as a PIL Image
        
    except Exception as e:
        print(f"An error occurred in create_hue_mask: {e}")
        return None
    
def create_saturation_mask(hsv_image_object, s_min):
    """Creates a mask based on a minimum Saturation threshold."""
    try:
        hsv_cv_image = np.array(hsv_image_object)
        
        # We only care about Saturation, so H and V ranges are 0-255
        lower_bound = np.array([0, s_min, 0])
        upper_bound = np.array([255, 255, 255])
        
        mask = cv2.inRange(hsv_cv_image, lower_bound, upper_bound)
        return Image.fromarray(mask) # Return as a PIL Image
        
    except Exception as e:
        print(f"An error occurred in create_saturation_mask: {e}")
        return None
    
# ================ LAB Space A And B Channels Analysis ================
def create_lab_ab_mask(lab_image_object, ab_max):
    """
    Creates a mask for pixels where both 'a' and 'b' channels are below a threshold.
    """
    try:
        lab_cv_image = np.array(lab_image_object)
        
        # We only care that 'a' and 'b' are low. L can be anything.
        # Channels are L, a, b.
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([255, ab_max, ab_max])
        
        mask = cv2.inRange(lab_cv_image, lower_bound, upper_bound)
        return Image.fromarray(mask)
        
    except Exception as e:
        print(f"An error occurred in create_lab_ab_mask: {e}")
        return None
    

# ==============================================================================
# ==================== END OF COLOR SPACES ====================
# ==============================================================================


# ==============================================================================
# ==================== MORPHOLOGICAL OPERATIONS ====================

def morph_closing(binary_mask_object, kernel_size=5):
    """
    Applies a morphological closing operation to a binary mask.
    
    Args:
        binary_mask_object (PIL.Image.Image): A binary mask.
        kernel_size (int): The size of the kernel for the operation.

    Returns:
        PIL.Image.Image: The mask after the closing operation.
    """
    try:
        mask = np.array(binary_mask_object.convert('L'))
        
        # Create the kernel for the morphological operation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply the closing operation
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


# ==============================================================================
# ==================== KEEP LARGEST CONTOUR ====================
    
def keep_largest_contour(binary_mask_object):
    """
    Finds all contours in a binary mask and returns a new mask
    containing only the one with the largest area.
    """
    try:
        mask = np.array(binary_mask_object.convert('L'))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a new blank mask to draw the largest contour on
            new_mask = np.zeros_like(mask)
            cv2.drawContours(new_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
            
            return Image.fromarray(new_mask)
        else:
            # Return the original mask if no contours are found
            return binary_mask_object
            
    except Exception as e:
        print(f"An error occurred while finding the largest contour: {e}")
        return None
    
    