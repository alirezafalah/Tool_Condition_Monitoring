import os
import time
from PIL import Image
import numpy as np
from .utils.optimized_processing import generate_masks, print_optimization_header

def run(config):
    """
    Processes pre-blurred images by combining background subtraction and
    multi-channel color masking. Uses pipeline-wide optimization method.
    """
    # --- Guardrail: Check if any masking is enabled ---
    method = config.get('BACKGROUND_SUBTRACTION_METHOD', 'none').lower()
    use_mc_mask = config.get('APPLY_MULTICHANNEL_MASK', False)
    if method == 'none' and not use_mc_mask:
        print("Warning: All masking methods are disabled. Exiting.")
        return

    input_dir = config['BLURRED_DIR']
    output_dir = config['FINAL_MASKS_DIR']
    optimization_method = config.get('OPTIMIZATION_METHOD', 'gpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No blurred images found in '{input_dir}'. Skipping.")
            return
    except FileNotFoundError:
        print(f"Error: Blurred data directory not found at '{input_dir}'.")
        return

    # Display optimization info
    print_optimization_header(optimization_method, f"Step 2: Mask Generation ({len(image_files)} images)")
    
    if method in ['absdiff', 'lab']:
        print(f"   Background Method: {method}")
    if use_mc_mask:
        print(f"   Multi-channel Mask: Enabled")

    # Run optimized mask generation
    start_time = time.time()
    
    success_count, failed_files = generate_masks(
        image_files=image_files,
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
        method=optimization_method
    )
    
    duration = time.time() - start_time
    
    # Print summary
    print(f"\n✅ Mask generation complete: {success_count}/{len(image_files)} images in {duration:.2f}s")
    print(f"📊 Throughput: {len(image_files)/duration:.2f} images/second")
    if failed_files:
        print(f"❌ Failed: {len(failed_files)} images")
        for error in failed_files[:5]:
            print(f"   - {error}")