from PIL import Image, ImageFilter

def apply_median_blur(tiff_path, kernel_size=15):
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
    except FileNotFoundError:
        print(f"Error: Input file not found at '{tiff_path}'")
        return None
    except Exception as e:
        print(f"An error occurred during blurring: {e}")
        return None