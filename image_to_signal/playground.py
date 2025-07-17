from PIL import Image, ImageFilter
# Assuming your functions are in these files
from image_utils import display_images, convert_tiff_to_jpeg
from filters import apply_median_blur, convert_to_hsv

if __name__ == "__main__":
    input_file = 'good_tool.tiff'
    kernel_size = 13

    # --- SETUP ---
    original_image = Image.open(input_file)
    
    # --- PATH A: Blur -> Convert to HSV ---
    blurred_image = apply_median_blur(input_file, kernel_size=kernel_size)
    path_a_result = convert_to_hsv(blurred_image)

    # --- PATH B: Convert to HSV -> Blur ---
    hsv_image = convert_to_hsv(original_image)
    path_b_result = hsv_image.filter(ImageFilter.MedianFilter(size=kernel_size))


    # --- VISUALIZE ---
    print("Showing Comparison: (Blur -> HSV) vs. (HSV -> Blur)...")
    if path_a_result and path_b_result:
        display_images([
            (path_a_result, 'Path A: Blur -> HSV'),
            (path_b_result, 'Path B: HSV -> Blur')
        ])
