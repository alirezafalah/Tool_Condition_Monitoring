import os
from PIL import Image
from tqdm import tqdm
from utils.filters import apply_median_blur

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

    # --- Part 1: Rename Raw Files ---
    # Check if the first file seems to be already renamed to avoid re-running
    if '_degrees.tiff' not in image_files[0]:
        print("Renaming raw files to include their rotation angle...")
        angle_step = 366.0 / config['images_for_366_deg']
        renamed_files = []
        for i, filename in enumerate(tqdm(image_files, desc="Renaming Raw Files")):
            current_angle = i * angle_step
            new_filename = f"{current_angle:07.2f}_degrees.tiff"
            
            old_path = os.path.join(raw_dir, filename)
            new_path = os.path.join(raw_dir, new_filename)
            
            try:
                os.rename(old_path, new_path)
                renamed_files.append(new_filename)
            except OSError as e:
                print(f"\nError renaming {filename}: {e}")
                continue
        # Update image_files to use the new names for the next step
        image_files = sorted(renamed_files)
        print("File renaming complete.")
    else:
        print("Raw files appear to be already renamed. Skipping renaming.")

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
