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