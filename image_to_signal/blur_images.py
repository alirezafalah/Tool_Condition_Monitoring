import os
from PIL import Image
from tqdm import tqdm
from filters import apply_median_blur

def main():
    # --- Configuration ---
    RAW_DIR = 'data/tool014gain10paperBG'
    PROCESSED_DIR = 'data/tool014gain10paperBG_blurred' # New output folder
    BLUR_KERNEL = 13
    
    # --- Main Logic ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    image_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(('.tiff', '.tif'))])

    print(f"Starting pre-processing: Applying blur to {len(image_files)} images...")
    for filename in tqdm(image_files, desc="Blurring Images"):
        input_path = os.path.join(RAW_DIR, filename)
        output_path = os.path.join(PROCESSED_DIR, filename)
        
        # Apply blur and save the result as a TIFF to preserve quality
        blurred_image = apply_median_blur(input_path, kernel_size=BLUR_KERNEL)
        if blurred_image:
            blurred_image.save(output_path, 'TIFF')
            
    print(f"\nPre-processing complete. Blurred images saved to '{PROCESSED_DIR}'")

if __name__ == "__main__":
    main()