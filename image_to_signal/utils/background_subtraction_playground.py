import os
import cv2
import numpy as np
from PIL import Image

# --- Local Imports from your project structure ---
# Make sure this script is in the 'utils' folder
from ..main import CONFIG
from .image_utils import display_images
from .filters import fill_holes, keep_largest_contour # Import filters for future use

def run_playground():
    """
    A sandbox for testing background subtraction on a sample image.
    """
    print("--- Starting Background Subtraction Playground ---")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # CONTROL PANEL
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 1. Define the path to your background image
    BACKGROUND_IMAGE_PATH = 'DATA/backgrounds/background.tiff'

    # 2. Define the threshold for creating the binary mask after subtraction
    #    A lower value is more sensitive to small differences.
    DIFFERENCE_THRESHOLD = 15 # Value from 0-255

    # 3. Choose a sample image to test on (0 is the first image)
    SAMPLE_IMAGE_INDEX = 117
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    # --- 1. Load Images ---
    try:
        # Load the background image
        print(f"Loading background image: {BACKGROUND_IMAGE_PATH}")
        background_pil = Image.open(BACKGROUND_IMAGE_PATH)

        # Load a sample blurred image from the directory specified in CONFIG
        blurred_dir = CONFIG['BLURRED_DIR']
        sample_files = sorted([f for f in os.listdir(blurred_dir) if f.endswith(('.tiff', '.tif'))])
        if not sample_files:
            print(f"Error: No images found in '{blurred_dir}'")
            return
        
        sample_image_name = sample_files[SAMPLE_IMAGE_INDEX]
        sample_image_path = os.path.join(blurred_dir, sample_image_name)
        print(f"Loading sample image: {sample_image_name}")
        sample_pil = Image.open(sample_image_path)

    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return

    # --- 2. Prepare for Subtraction ---
    # Convert both images to grayscale NumPy arrays for subtraction
    background_gray = cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2GRAY)
    sample_gray = cv2.cvtColor(np.array(sample_pil), cv2.COLOR_RGB2GRAY)

    # --- 3. Perform Background Subtraction ---
    # Calculate the absolute difference between the two images
    diff_image = cv2.absdiff(sample_gray, background_gray)
    print("Calculated absolute difference between images.")

    # --- 4. Create a Binary Mask ---
    # Use a threshold to turn the difference image into a binary mask.
    # Pixels with a difference greater than the threshold become white (255).
    _, binary_mask = cv2.threshold(diff_image, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)
    print(f"Created binary mask with threshold: {DIFFERENCE_THRESHOLD}")
    
    # Convert the final NumPy mask to a PIL Image for display
    mask_pil = Image.fromarray(binary_mask)

    # --- 5. Display the Results ---
    display_images([
        (sample_pil, 'Original Blurred Image'),
        (background_pil, 'Background Image'),
        (mask_pil, 'Subtracted Mask')
    ])

    # --- 6. (Optional) Post-Processing ---
    # You can uncomment these lines to see the effect of your filters
    # print("\nApplying post-processing filters...")
    # filled = fill_holes(mask_pil)
    # largest_contour = keep_largest_contour(filled)
    #
    # display_images([
    #     (mask_pil, 'Initial Mask'),
    #     (filled, 'Mask with Holes Filled'),
    #     (largest_contour, 'Final Largest Contour')
    # ])

    print("\n--- Playground Script Finished ---")


if __name__ == "__main__":
    run_playground()
