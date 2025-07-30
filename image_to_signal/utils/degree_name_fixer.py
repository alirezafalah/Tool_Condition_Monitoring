import os
from tqdm import tqdm

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# !! IMPORTANT !!
# You will need to run this script TWICE:
# 1. Once for your RAW image directory.
# 2. Once for your BLURRED image directory.
#
# !! SET THE DIRECTORY TO FIX BELOW !!
TARGET_DIR = '../data/subtract2' 

# The total degrees for a full rotation
TOTAL_DEGREES = 360.0
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def main():
    """
    A temporary, one-time script to correct the angles in filenames
    based on a new calculation.
    """
    print(f"--- Filename Correction Script ---")
    print(f"Targeting directory: '{TARGET_DIR}'")

    try:
        image_files = sorted([f for f in os.listdir(TARGET_DIR) if f.endswith(('.tiff', '.tif', '.jpg', '.jpeg'))])
        if not image_files:
            print("No image files found to rename.")
            return
        
        num_files = len(image_files)
        angle_step = TOTAL_DEGREES / num_files
        print(f"Found {num_files} files. New angle step will be {angle_step:.4f} degrees.")

    except FileNotFoundError:
        print(f"Error: Target directory not found at '{TARGET_DIR}'. Please check the path.")
        return

    # Loop through the files and rename them with the corrected angle
    for i, filename in enumerate(tqdm(image_files, desc="Correcting Filenames")):
        correct_angle = i * angle_step
        # Format with leading zeros for correct alphabetical sorting
        new_filename = f"{correct_angle:07.2f}_degrees.jpg"
        
        old_path = os.path.join(TARGET_DIR, filename)
        new_path = os.path.join(TARGET_DIR, new_filename)
        
        # To avoid errors if the new name is the same as the old, we check first
        if old_path != new_path:
            try:
                os.rename(old_path, new_path)
            except OSError as e:
                print(f"\nError renaming {filename}: {e}")
                continue
    
    print("\nFilename correction complete.")

if __name__ == "__main__":
    main()
