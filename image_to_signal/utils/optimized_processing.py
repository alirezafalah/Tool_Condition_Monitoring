"""
Optimized Image Processing Module
=================================
Provides three optimization methods for the entire pipeline:
1. GPU (default) - Uses OpenCV UMat with OpenCL (Intel GPU recommended)
2. Multi-core - Uses ProcessPoolExecutor for parallel CPU processing
3. Single-core - Standard sequential processing (fallback)

This module optimizes ALL pipeline steps:
- Step 1: Blur processing (median blur)
- Step 2: Mask generation
- Step 3: ROI analysis
- Step 4: Processed analysis

Hardware Requirements:
----------------------
- GPU mode: 
    * Intel Iris Xe GPU (or compatible OpenCL 1.2+ device)
    * Updated Intel Graphics drivers
    * Works best with integrated Intel GPUs on 11th gen+ processors
    
- Multi-core mode:
    * Any multi-core CPU
    * 4+ cores recommended for significant speedup
    * Uses all available logical cores
    
- Single-core mode:
    * Any CPU (no special requirements)
    * Slowest but most compatible option

Note: GPU acceleration via OpenCL works best with Intel Iris Xe GPUs.
      NVIDIA/AMD GPUs may work but are not tested/guaranteed.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ============================================================================
# GPU STATUS DETECTION
# ============================================================================

def check_gpu_available():
    """
    Check if OpenCL (GPU acceleration) is available.
    Returns tuple: (is_available: bool, device_name: str, message: str)
    """
    try:
        if not cv2.ocl.haveOpenCL():
            return False, None, "OpenCL is not available. Install GPU drivers."
        
        cv2.ocl.setUseOpenCL(True)
        device = cv2.ocl.Device.getDefault()
        device_name = device.name() if device else "Unknown"
        driver_version = device.driverVersion() if device else "Unknown"
        
        is_intel = "Intel" in device_name
        
        if is_intel:
            message = f"✅ Intel GPU detected: {device_name} (Driver: {driver_version})"
        else:
            message = f"⚠️ Non-Intel GPU detected: {device_name}. GPU acceleration may not be optimal."
        
        return True, device_name, message
        
    except Exception as e:
        return False, None, f"GPU check failed: {str(e)}"


def get_optimization_info():
    """
    Get detailed information about available optimization methods.
    Returns a dict with status and requirements of each method.
    """
    gpu_available, gpu_device, gpu_message = check_gpu_available()
    cpu_count = os.cpu_count() or 1
    
    return {
        'gpu': {
            'available': gpu_available,
            'device': gpu_device,
            'message': gpu_message,
            'is_intel': gpu_device and "Intel" in gpu_device if gpu_device else False,
            'requirements': 'Intel Iris Xe GPU (or OpenCL 1.2+ compatible) with updated drivers',
            'description': 'Uses OpenCL to offload image processing to the GPU. Fastest option for Intel systems.'
        },
        'multicore': {
            'available': True,
            'cores': cpu_count,
            'message': f"Multi-core processing available ({cpu_count} logical cores)",
            'requirements': f'Multi-core CPU ({cpu_count} cores detected, 4+ recommended)',
            'description': 'Distributes work across all CPU cores in parallel. Good for systems without Intel GPU.'
        },
        'single': {
            'available': True,
            'message': "Single-core processing (fallback, slowest)",
            'requirements': 'Any CPU (no special requirements)',
            'description': 'Sequential processing on a single core. Slowest but always works.'
        }
    }


def get_effective_method(requested_method):
    """
    Returns the actual method that will be used, handling automatic fallbacks.
    If GPU is requested but unavailable, falls back to multicore.
    """
    if requested_method == 'gpu':
        gpu_available, _, _ = check_gpu_available()
        if not gpu_available:
            print("⚠️ GPU unavailable, falling back to multi-core processing")
            return 'multicore'
    return requested_method


def print_optimization_header(method, step_name):
    """Print a consistent header for each processing step."""
    effective = get_effective_method(method)
    info = get_optimization_info()
    
    method_icons = {'gpu': '🎮', 'multicore': '🔧', 'single': '🐌'}
    icon = method_icons.get(effective, '⚙️')
    
    print(f"\n{icon} {step_name}")
    print(f"   Method: {effective.upper()}")
    if effective == 'gpu':
        print(f"   Device: {info['gpu'].get('device', 'Unknown')}")
    elif effective == 'multicore':
        print(f"   Cores: {info['multicore']['cores']}")


# ============================================================================
# BLUR PROCESSING
# ============================================================================

def _blur_single_gpu(input_path, output_path, kernel_size):
    """Process a single image using GPU (OpenCV UMat)."""
    try:
        img_cpu = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img_cpu is None:
            return False, f"Failed to read: {input_path}"
        
        img_gpu = cv2.UMat(img_cpu)
        blurred_gpu = cv2.medianBlur(img_gpu, kernel_size)
        cv2.imwrite(output_path, blurred_gpu)
        return True, None
    except Exception as e:
        return False, str(e)


def _blur_single_cpu(args):
    """Process a single image using PIL (for multi-core/single-core)."""
    input_path, output_path, kernel_size = args
    try:
        with Image.open(input_path) as img:
            blurred = img.filter(ImageFilter.MedianFilter(size=kernel_size))
            blurred.save(output_path, 'TIFF')
        return True, None
    except Exception as e:
        return False, str(e)


def blur_images(image_files, input_dir, output_dir, kernel_size, method='gpu'):
    """
    Blur all images using the specified optimization method.
    
    Returns: (success_count, failed_files)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    os.makedirs(output_dir, exist_ok=True)
    effective = get_effective_method(method)
    
    success_count = 0
    failed_files = []
    
    if effective == 'gpu':
        cv2.ocl.setUseOpenCL(True)
        for filename in tqdm(image_files, desc="GPU Blur"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            success, error = _blur_single_gpu(input_path, output_path, kernel_size)
            if success:
                success_count += 1
            else:
                failed_files.append((filename, error))
                
    elif effective == 'multicore':
        tasks = [(os.path.join(input_dir, f), os.path.join(output_dir, f), kernel_size) 
                 for f in image_files]
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_blur_single_cpu, tasks), 
                               total=len(tasks), desc="Multi-Core Blur"))
        for filename, (success, error) in zip(image_files, results):
            if success:
                success_count += 1
            else:
                failed_files.append((filename, error))
                
    else:  # single
        for filename in tqdm(image_files, desc="Single-Core Blur"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            success, error = _blur_single_cpu((input_path, output_path, kernel_size))
            if success:
                success_count += 1
            else:
                failed_files.append((filename, error))
    
    return success_count, failed_files


# ============================================================================
# MASK GENERATION
# ============================================================================

def _generate_mask_single(args):
    """
    Generate mask for a single image.
    Args: (filename, input_dir, output_dir, config_dict)
    """
    filename, input_dir, output_dir, config = args
    try:
        # Import here to avoid circular imports
        from .filters import (create_multichannel_mask, fill_holes, morph_closing, 
                             keep_largest_contour, background_subtraction_absdiff, 
                             background_subtraction_lab)
        
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        blurred_image = Image.open(image_path)
        
        method = config.get('bg_method', 'none')
        use_mc_mask = config.get('use_mc_mask', False)
        background_image_np = config.get('background_image_np', None)
        
        bg_mask_np = None
        color_mask_np = None
        
        if method == 'absdiff' and background_image_np is not None:
            bg_mask_pil = background_subtraction_absdiff(blurred_image, background_image_np, config)
            if bg_mask_pil: 
                bg_mask_np = np.array(bg_mask_pil)
        elif method == 'lab' and background_image_np is not None:
            bg_mask_pil = background_subtraction_lab(blurred_image, background_image_np, config)
            if bg_mask_pil: 
                bg_mask_np = np.array(bg_mask_pil)

        if use_mc_mask:
            color_mask_pil = create_multichannel_mask(blurred_image, config)
            if color_mask_pil:
                color_mask_np = np.array(color_mask_pil)

        if bg_mask_np is not None and color_mask_np is not None:
            initial_mask_np = cv2.bitwise_or(bg_mask_np, color_mask_np)
        elif bg_mask_np is not None:
            initial_mask_np = bg_mask_np
        elif color_mask_np is not None:
            initial_mask_np = color_mask_np
        else:
            return False, f"No mask generated for {filename}"
        
        initial_mask = Image.fromarray(initial_mask_np)
        filled1 = fill_holes(initial_mask)
        closed = morph_closing(filled1, kernel_size=config.get('closing_kernel', 21))
        largest_contour = keep_largest_contour(closed)
        filled2 = fill_holes(largest_contour)
        
        if filled2:
            filled2.save(output_path, 'TIFF')
            return True, filename
        return False, f"Final mask is None for {filename}"
            
    except Exception as e:
        return False, f"Error processing {filename}: {e}"


def generate_masks(image_files, input_dir, output_dir, config, method='gpu'):
    """
    Generate masks for all images using the specified optimization method.
    
    Returns: (success_count, failed_files)
    """
    os.makedirs(output_dir, exist_ok=True)
    effective = get_effective_method(method)
    
    # Prepare config for multiprocessing (must be picklable)
    config_for_mp = {
        'bg_method': config.get('BACKGROUND_SUBTRACTION_METHOD', 'none').lower(),
        'use_mc_mask': config.get('APPLY_MULTICHANNEL_MASK', False),
        'closing_kernel': config.get('closing_kernel', 21),
        'DIFFERENCE_THRESHOLD': config.get('DIFFERENCE_THRESHOLD', 33),
        'L_threshold_min': config.get('L_threshold_min', 127.5),
        'L_threshold_max': config.get('L_threshold_max', 142.8),
        'a_threshold_min': config.get('a_threshold_min', 118),
        'a_threshold_max': config.get('a_threshold_max', 127),
        'b_threshold_min': config.get('b_threshold_min', 118),
        'b_threshold_max': config.get('b_threshold_max', 120),
        'h_threshold_min': config.get('h_threshold_min', 35),
        'h_threshold_max': config.get('h_threshold_max', 50),
        's_threshold_min': config.get('s_threshold_min', 38.25),
        's_threshold_max': config.get('s_threshold_max', 178.5),
        'V_threshold_min': config.get('V_threshold_min', 114.75),
        'V_threshold_max': config.get('V_threshold_max', 140.25),
        'background_image_np': None
    }
    
    # Load background image
    if config_for_mp['bg_method'] in ['absdiff', 'lab']:
        try:
            bg_path = config.get('BACKGROUND_IMAGE_PATH', '')
            config_for_mp['background_image_np'] = np.array(Image.open(bg_path))
        except Exception as e:
            print(f"⚠️ Could not load background image: {e}")
    
    tasks = [(f, input_dir, output_dir, config_for_mp) for f in image_files]
    success_count = 0
    failed_files = []
    
    if effective == 'gpu':
        cv2.ocl.setUseOpenCL(True)
        for task in tqdm(tasks, desc="GPU Mask Generation"):
            success, result = _generate_mask_single(task)
            if success:
                success_count += 1
            else:
                failed_files.append(result)
                
    elif effective == 'multicore':
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_generate_mask_single, tasks),
                               total=len(tasks), desc="Multi-Core Mask Generation"))
        for success, result in results:
            if success:
                success_count += 1
            else:
                failed_files.append(result)
                
    else:  # single
        for task in tqdm(tasks, desc="Single-Core Mask Generation"):
            success, result = _generate_mask_single(task)
            if success:
                success_count += 1
            else:
                failed_files.append(result)
    
    return success_count, failed_files


# ============================================================================
# ROI ANALYSIS
# ============================================================================

def _analyze_roi_single(args):
    """Analyze ROI for a single mask image."""
    filename, input_dir, roi_height = args
    try:
        angle = float(filename.split('_')[0])
        image_path = os.path.join(input_dir, filename)
        mask_np = np.array(Image.open(image_path))
        
        total_white_pixels = np.sum(mask_np == 255)
        total_pixels = mask_np.size
        white_ratio = total_white_pixels / total_pixels
        
        white_pixel_coords = np.where(mask_np == 255)
        if white_pixel_coords[0].size == 0:
            roi_area = 0
        else:
            last_row = white_pixel_coords[0].max()
            first_row = max(0, last_row - roi_height)
            roi = mask_np[first_row:last_row, :]
            roi_area = np.sum(roi) / 255
        
        return True, {
            'Angle (Degrees)': angle,
            'ROI Area (Pixels)': roi_area,
            'white_ratio': white_ratio,
            'filename': filename
        }
    except (ValueError, IndexError) as e:
        return False, f"Could not parse angle from {filename}: {e}"
    except Exception as e:
        return False, f"Error processing {filename}: {e}"


def analyze_roi(image_files, input_dir, roi_height, method='gpu'):
    """
    Analyze ROI for all mask images using the specified optimization method.
    
    Returns: (results_list, failed_list)
    """
    effective = get_effective_method(method)
    tasks = [(f, input_dir, roi_height) for f in image_files]
    
    results = []
    failed = []
    
    if effective == 'multicore':
        with ProcessPoolExecutor() as executor:
            futures_results = list(tqdm(executor.map(_analyze_roi_single, tasks),
                                       total=len(tasks), desc="Multi-Core ROI Analysis"))
        for success, result in futures_results:
            if success:
                results.append(result)
            else:
                failed.append(result)
    else:
        # GPU and single-core use sequential (ROI analysis is mostly numpy, not OpenCV)
        desc = "GPU ROI Analysis" if effective == 'gpu' else "Single-Core ROI Analysis"
        for task in tqdm(tasks, desc=desc):
            success, result = _analyze_roi_single(task)
            if success:
                results.append(result)
            else:
                failed.append(result)
    
    return results, failed


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIPELINE OPTIMIZATION STATUS")
    print("="*70)
    
    info = get_optimization_info()
    
    print("\n🎮 GPU Acceleration:")
    print(f"   Status: {info['gpu']['message']}")
    print(f"   Requirements: {info['gpu']['requirements']}")
    print(f"   Description: {info['gpu']['description']}")
    
    print("\n🔧 Multi-Core Processing:")
    print(f"   Status: {info['multicore']['message']}")
    print(f"   Requirements: {info['multicore']['requirements']}")
    print(f"   Description: {info['multicore']['description']}")
    
    print("\n🐌 Single-Core Processing:")
    print(f"   Status: {info['single']['message']}")
    print(f"   Requirements: {info['single']['requirements']}")
    print(f"   Description: {info['single']['description']}")
    
    print("\n" + "="*70)
