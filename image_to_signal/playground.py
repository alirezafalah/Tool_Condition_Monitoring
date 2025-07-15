from image_utils import display_images
from filters import apply_median_blur
from PIL import Image

if __name__ == "__main__":
    input_file = 'attached_chip.tiff'

    # 1. Open the original image to get its object
    original_image_object = Image.open(input_file)
    
    # 2. Apply the median blur and get the blurred image object
    blurred_image_object = apply_median_blur(input_file, kernel_size=5)

    # 3. Display both images side-by-side if the blur was successful
    if original_image_object and blurred_image_object:
        display_images([
            (original_image_object, 'Original Image'),
            (blurred_image_object, 'Median Blur (k=15)')
        ])