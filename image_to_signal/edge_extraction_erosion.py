import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_edges_via_erosion(config):
    """
    Applies morphological erosion to a set of binary masks to extract the
    outer edge of the main object.

    The process is:
    1. Read a binary image (the silhouette).
    2. Create an "eroded" version by shrinking the silhouette inwards.
    3. Subtract the eroded image from the original to get the edge.
    4. Save the resulting edge image.
    """
    # --- Setup ---
    input_dir = config['INPUT_DIR']
    output_dir = config['OUTPUT_DIR']
    erosion_kernel_size = config['EROSION_KERNEL_SIZE']
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Image Processing ---
    try:
        # Get a sorted list of image files to process
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
        if not image_files:
            print(f"Error: No images found in the input directory '{input_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'.")
        return

    print(f"Extracting edges from {len(image_files)} images...")

    # Define the erosion kernel. A larger kernel means a thicker edge.
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)

    for filename in tqdm(image_files, desc="Extracting Edges"):
        try:
            image_path = os.path.join(input_dir, filename)
            
            # Read the original binary image (as grayscale)
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue

            # 1. Apply the erosion operation
            eroded_img = cv2.erode(original_img, kernel, iterations=1)

            # 2. Subtract the eroded image from the original to get the edge
            # The result will have white pixels (255) only where the edge was.
            edge_img = cv2.subtract(original_img, eroded_img)

            # 3. Save the resulting edge image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, edge_img)

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    print(f"\nEdge extraction complete. Results saved in '{output_dir}'.")


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        # --- Input Directory ---
        # This should contain the binary silhouette masks.
        # The path is relative to the `image_to_signal` directory.
        'INPUT_DIR': 'data/tool014gain10paperBG_final_masks',
        
        # --- Output Directory ---
        # A new folder where the resulting edge images will be saved.
        'OUTPUT_DIR': 'data/tool014gain10paperBG_erroded_edges',
        
        # --- Erosion Parameter ---
        # This controls the thickness of the resulting edge.
        # A value of 3 is a good starting point. Increase for a thicker edge.
        'EROSION_KERNEL_SIZE': 51
    }
    
    # --- Run the script ---
    extract_edges_via_erosion(config)
