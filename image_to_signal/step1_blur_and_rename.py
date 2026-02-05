import os
import time
from .utils.optimized_processing import blur_images, get_optimization_info, print_optimization_header

def run(config):
    """
    Applies blur to raw images and saves them.
    Uses the pipeline-wide optimization method from config['OPTIMIZATION_METHOD'].
    """
    raw_dir = config['RAW_DIR']
    blurred_dir = config['BLURRED_DIR']
    kernel_size = config.get('blur_kernel', 13)
    optimization_method = config.get('OPTIMIZATION_METHOD', 'gpu')
    
    os.makedirs(blurred_dir, exist_ok=True)

    try:
        image_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No raw images found in '{raw_dir}'. Skipping.")
            return
    except FileNotFoundError:
        print(f"Error: Raw data directory not found at '{raw_dir}'.")
        return

    print("Using existing raw filenames (renaming handled by separate script).")

    # Check if blurred images already exist
    existing_blurred = [f for f in os.listdir(blurred_dir) if f.endswith(('.tiff', '.tif'))]
    if len(existing_blurred) >= len(image_files):
        print(f"Blurred images already exist in '{blurred_dir}'. Skipping blur step.")
        return

    # Display optimization info
    print_optimization_header(optimization_method, f"Step 1: Blur Processing ({len(image_files)} images)")

    # Run the blur processing
    start_time = time.time()
    
    success_count, failed_files = blur_images(
        image_files=image_files,
        input_dir=raw_dir,
        output_dir=blurred_dir,
        kernel_size=kernel_size,
        method=optimization_method
    )
    
    duration = time.time() - start_time
    
    # Print summary
    print(f"\n✅ Blur complete: {success_count}/{len(image_files)} images in {duration:.2f}s")
    print(f"📊 Throughput: {len(image_files)/duration:.2f} images/second")
    if failed_files:
        print(f"❌ Failed: {len(failed_files)} images")
        for fname, error in failed_files[:5]:
            print(f"   - {fname}: {error}")
