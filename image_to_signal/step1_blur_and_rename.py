import os
from PIL import Image
from tqdm import tqdm
from .utils.filters import apply_median_blur

def run(config):
    """
    First, renames raw images to include their angle, then applies blur and saves them.
    This is the slowest step and is designed to be run only once.
    """
    raw_dir = config['RAW_DIR']
    blurred_dir = config['BLURRED_DIR']
    
    os.makedirs(blurred_dir, exist_ok=True)

    try:
        image_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No raw images found in '{raw_dir}'. Skipping.")
            return
    except FileNotFoundError:
        print(f"Error: Raw data directory not found at '{raw_dir}'.")
        return

    # Renaming removed; use external script after determining 360 frame count.
    print("Using existing raw filenames (renaming handled by separate script).")

    # --- Part 2: Blur Renamed Files ---
    # Check if blurred images already exist
    if len(os.listdir(blurred_dir)) >= len(image_files):
        print(f"Blurred images already exist in '{blurred_dir}'. Skipping blur step.")
        return

    print(f"Applying blur to {len(image_files)} renamed images...")
    for filename in tqdm(image_files, desc="Blurring Images"):
        # The input is now the renamed file from the raw directory
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(blurred_dir, filename)
        
        # Apply blur
        blurred_image = apply_median_blur(input_path, kernel_size=config['blur_kernel'])
        
        # Save the result with the same angle-based name
        if blurred_image:
            blurred_image.save(output_path, 'TIFF')
