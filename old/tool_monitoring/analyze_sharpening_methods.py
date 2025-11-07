import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, restoration, morphology
from scipy import ndimage
from pathlib import Path

# Path to the directory containing tool images
tool_dir = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\ENDMILL-E6-001\TOOL"

# Set output directory to a "comparision" folder in the ENDMILL-E6-001 directory
comparision_dir = os.path.join(os.path.dirname(tool_dir), "comparision")
os.makedirs(comparision_dir, exist_ok=True)
print(f"Output directory: {comparision_dir}")

# Create a directory for plots in the comparision folder
plots_dir = os.path.join(comparision_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Define the region of interest (ROI) for cropping
roi_x1, roi_y1, roi_x2, roi_y2 = 180, 580, 300, 732

def load_image(image_path):
    """Load an image from the given path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    # Convert BGR to RGB for better visualization
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop_to_roi(image):
    """Crop the image to the specified region of interest."""
    return image[roi_y1:roi_y2, roi_x1:roi_x2]

def save_visualization(original, results, titles, filename):
    """
    Save a figure with original image and processed results without white space.
    
    Args:
        original: Original image
        results: List of processed images
        titles: List of titles for each processed image
        filename: Output filename
    """
    # Crop images to ROI
    original_cropped = crop_to_roi(original)
    results_cropped = [crop_to_roi(img) for img in results]
    
    n_images = len(results_cropped) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
    
    # If there's only one image to display, axes won't be an array
    if n_images == 2:
        axes = [axes[0], axes[1]]
    
    axes[0].imshow(original_cropped)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i, (result, title) in enumerate(zip(results_cropped, titles)):
        axes[i+1].imshow(result)
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
        
    # Remove white space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    
    # Save to comparision_dir
    plt.savefig(os.path.join(comparision_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_all_visualization(original, results, titles, filename):
    """Save a figure with original and all processed images without white space."""
    # Crop images to ROI
    original_cropped = crop_to_roi(original)
    results_cropped = [crop_to_roi(img) for img in results]
    
    n_cols = 4  # Number of columns
    n_rows = (len(results_cropped) + 1 + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_cropped)
    plt.title("Original")
    plt.axis('off')
    
    # Plot all result images
    for i, (result, title) in enumerate(zip(results_cropped, titles)):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(result)
        plt.title(title)
        plt.axis('off')
    
    # Remove white space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    
    # Save to comparision_dir
    plt.savefig(os.path.join(comparision_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_image(img, filename):
    """Save an image to the comparision directory."""
    # Crop to ROI
    img_cropped = crop_to_roi(img)
    
    # Convert RGB to BGR for OpenCV
    if len(img_cropped.shape) == 3 and img_cropped.shape[2] == 3:
        img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(comparision_dir, filename), img_cropped)

# Method 1: Basic Sharpening using Unsharp Mask
def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Apply unsharp masking to enhance image details."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
        
    return sharpened

# Method 2: Adaptive Histogram Equalization (CLAHE)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(lab))
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result

# Method 3: Deconvolution (Wiener filter)
def wiener_deconvolution(image, psf_size=5, noise_power=0.01):
    """Apply Wiener deconvolution to deblur the image."""
    # Create a simple point spread function (PSF)
    psf = np.ones((psf_size, psf_size)) / (psf_size ** 2)
    
    # Apply Wiener deconvolution to each channel
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        deconvolved = restoration.wiener(image[:, :, i].astype(np.float32) / 255, 
                                        psf, 
                                        noise_power)
        # Clip values to [0, 1] range
        result[:, :, i] = np.clip(deconvolved, 0, 1)
    
    # Convert back to uint8
    result = (result * 255).astype(np.uint8)
    return result

# Method 4: Edge Preservation using Bilateral Filter
def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Method 5: Detail Enhancement using Guided Filter
def guided_filter(image, radius=5, eps=0.1):
    """Apply guided filter for detail enhancement."""
    # Convert to grayscale for the guidance
    guidance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply guided filter to each channel
    enhanced = np.zeros_like(image)
    for i in range(3):
        enhanced[:, :, i] = cv2.ximgproc.guidedFilter(
            guidance, image[:, :, i], radius, eps)
    
    return enhanced

# Method 6: Sharpening with Laplacian
def laplacian_sharpening(image, kernel_size=3, scale=0.5):
    """Sharpen the image using Laplacian operator."""
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Apply Laplacian
    lap = cv2.Laplacian(img_float, cv2.CV_32F, ksize=kernel_size)
    
    # Add weighted Laplacian to the original image
    sharpened = img_float - scale * lap
    
    # Clip values to valid range and convert back to uint8
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# Method 7: High-pass Filtering
def high_pass_filter(image, sigma=1.0):
    """Apply high-pass filter to enhance details."""
    # Create low-pass filtered version
    low_pass = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        low_pass[:, :, i] = filters.gaussian(image[:, :, i], sigma=sigma)
    
    # Subtract low-pass from original to get high-pass
    high_pass = image.astype(np.float32) - low_pass
    
    # Add high-pass to original
    enhanced = image.astype(np.float32) + high_pass
    
    # Clip values to valid range and convert back to uint8
    return np.clip(enhanced, 0, 255).astype(np.uint8)

# Method 8: Normalize & Enhance Contrast
def normalize_and_enhance(image):
    """Normalize the image and enhance contrast."""
    # Normalize 
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        channel = image[:, :, i].astype(np.float32)
        if channel.std() > 0:  # Avoid division by zero
            normalized[:, :, i] = (channel - channel.mean()) / channel.std() * 64 + 128
    
    # Clip values and convert back to uint8
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    # Apply contrast stretching
    p2, p98 = np.percentile(normalized, (2, 98))
    enhanced = exposure.rescale_intensity(normalized, in_range=(p2, p98))
    
    return enhanced

# Method 9: Multi-scale Sharpening
def multi_scale_sharpening(image, scales=[3, 5, 9], weights=[0.5, 0.3, 0.2]):
    """Apply multi-scale sharpening using multiple unsharp masks."""
    result = image.astype(np.float32)
    
    for scale, weight in zip(scales, weights):
        kernel_size = (scale, scale)
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        detail = image.astype(np.float32) - blurred.astype(np.float32)
        result += weight * detail
    
    return np.clip(result, 0, 255).astype(np.uint8)

# Method 10: Background Removal preparation using thresholding
def prepare_for_background_removal(image):
    """Prepare the image for background removal by enhancing the contrast
    between the object and background."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply bilateral filter to preserve edges
    filtered = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    
    # Threshold to create a binary mask (tool vs background)
    _, threshold = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    
    # Create RGB representation of the mask for visualization
    mask_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    return mask_rgb

# Method 11: Edge Detection Enhancement
def edge_enhanced_preprocessing(image, sigma=1.0, low_threshold=10, high_threshold=60):
    """Enhance edges to improve boundary detection for background removal."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Canny edge detection
    edges = cv2.Canny(filtered, low_threshold, high_threshold)
    
    # Dilate edges to make them more visible
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create an edge-enhanced image by adding edges to original
    edge_enhanced = image.copy()
    edge_enhanced[dilated_edges > 0] = [255, 0, 0]  # Mark edges in red
    
    return edge_enhanced

# Method 12: K-means Segmentation
def kmeans_segmentation(image, k=3):
    """Apply K-means clustering for segmentation."""
    # Reshape image for k-means
    pixels = image.reshape((-1, 3)).astype(np.float32)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image shape
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

# Method 13: Adaptive Thresholding
def adaptive_threshold_preprocessing(image, block_size=11, c=2):
    """Apply adaptive thresholding to enhance local features."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )
    
    # Create RGB representation for visualization
    result = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
    
    return result

# Method 14: Watershed Segmentation
def watershed_segmentation(image):
    """Apply watershed algorithm for image segmentation."""
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)
    
    # Create result image with colored regions
    result = image.copy()
    result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    return result

# Method 15: Focus Enhancement (for blurry images)
def focus_enhancement(image, kernel_size=5, sharp_amount=2.0):
    """Enhance focus in blurry images."""
    # Convert to LAB color space (L channel controls luminance)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply unsharp mask to L channel
    gaussian = cv2.GaussianBlur(l, (kernel_size, kernel_size), 0)
    unsharp_mask = cv2.addWeighted(l, 1.0 + sharp_amount, gaussian, -sharp_amount, 0)
    
    # Merge channels and convert back to RGB
    enhanced_lab = cv2.merge([unsharp_mask, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def create_comparison_visualizations():
    """Create visual comparisons of all enhancement methods for a sample image."""
    # Path to the first image (tool001.jpg)
    image_path = os.path.join(tool_dir, "tool001.jpg")
    
    # Load the image
    original = load_image(image_path)
    
    # Define all methods with their names and parameters
    methods = [
        (unsharp_masking, "Unsharp Masking", {}),
        (apply_clahe, "CLAHE", {}),
        (wiener_deconvolution, "Wiener Deconvolution", {}),
        (bilateral_filter, "Bilateral Filter", {}),
        (guided_filter, "Guided Filter", {}),
        (laplacian_sharpening, "Laplacian Sharpening", {}),
        (high_pass_filter, "High-pass Filter", {}),
        (normalize_and_enhance, "Normalize & Enhance", {}),
        (multi_scale_sharpening, "Multi-scale Sharpening", {}),
        (prepare_for_background_removal, "Background Removal Prep", {}),
        (edge_enhanced_preprocessing, "Edge Enhanced", {}),
        (kmeans_segmentation, "K-means Segmentation", {}),
        (adaptive_threshold_preprocessing, "Adaptive Threshold", {}),
        (watershed_segmentation, "Watershed Segmentation", {}),
        (focus_enhancement, "Focus Enhancement", {})
    ]
    
    # Process each method and save individual results
    for method_func, method_name, params in methods:
        print(f"Applying {method_name} for visualization...")
        result = method_func(original, **params)
        save_image(result, f"tool001_{method_name.replace(' ', '_').lower()}.jpg")
        
        # Save side-by-side comparison
        save_visualization(original, [result], [method_name], 
                          f"tool001_{method_name.replace(' ', '_').lower()}_comparison.jpg")
    
    # Process and save all methods together for comparison
    print("Generating combined visualization...")
    results = []
    titles = []
    for method_func, method_name, params in methods:
        results.append(method_func(original, **params))
        titles.append(method_name)
    
    # Create figure with all methods
    save_all_visualization(original, results, titles, "tool001_all_methods_comparison.jpg")
    
    print("All comparison visualizations have been saved.")

def analyze_enhancement_methods():
    """Apply background subtraction to processed images and plot the results."""
    
    # First create visual comparisons to see what each method does
    create_comparison_visualizations()
    
    # Paths for the analysis
    base_dir = os.path.dirname(os.path.dirname(tool_dir))
    base_tool_dir = os.path.join(base_dir, "ENDMILL-E6-001", "TOOL")
    bg_dir = os.path.join(base_dir, "ENDMILL-E6-001", "BG")
    
    # Create a directory for each method's processed images
    methods = [
        "original",
        "unsharp_masking", 
        "clahe", 
        "wiener_deconvolution",
        "bilateral_filter", 
        "guided_filter",
        "laplacian_sharpening",
        "high_pass_filter",
        "normalize_and_enhance",
        "multi_scale_sharpening",
        "prepare_for_background_removal",
        "edge_enhanced_preprocessing",
        "kmeans_segmentation",
        "adaptive_threshold_preprocessing",
        "watershed_segmentation",
        "focus_enhancement"
    ]
    
    method_dirs = {}
    for method in methods:
        method_dir = os.path.join(comparision_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        method_dirs[method] = method_dir
    
    # Median background calculation
    def calculate_median_background(bg_dir):
        bg_frames = sorted(os.listdir(bg_dir))
        images = [cv2.imread(os.path.join(bg_dir, frame)) for frame in bg_frames]
        stacked_images = np.stack(images, axis=0)
        median_img = np.median(stacked_images, axis=0).astype(np.uint8)
        return median_img
    
    # Background subtraction and processing
    def process_with_method(image_path, median_bg, method_func=None, method_params=None, thresh_value=55, kernel_size=35):
        if method_params is None:
            method_params = {}
            
        image = cv2.imread(image_path)
        
        # Apply enhancement method if provided
        if method_func:
            # Convert to RGB for processing methods
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_rgb = method_func(image_rgb, **method_params)
            # Convert back to BGR for OpenCV
            image = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        
        # Background subtraction
        difference = cv2.absdiff(image, median_bg)
        grayscale_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(grayscale_diff, thresh_value, 255, cv2.THRESH_BINARY)
        
        # Apply closing operation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_img = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        
        return closed_img
    
    # Analyze white pixel count
    def analyze_white_pixels(image, offset=50):
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
            
        # Find largest contour and bottom-most point
        largest_contour = max(contours, key=cv2.contourArea)
        bottom_most_point = max(largest_contour, key=lambda pt: pt[0][1])
        bottom_most_row = bottom_most_point[0][1]
        
        # Define target row with offset
        target_row = max(0, bottom_most_row - offset)
        white_pixel_count = np.sum(image[target_row:, :] == 255)
        
        return white_pixel_count
    
    print("Calculating median background...")
    median_bg = calculate_median_background(bg_dir)
    
    # Map method names to functions
    method_funcs = {
        "original": None,
        "unsharp_masking": unsharp_masking,
        "clahe": apply_clahe,
        "wiener_deconvolution": wiener_deconvolution,
        "bilateral_filter": bilateral_filter,
        "guided_filter": guided_filter,
        "laplacian_sharpening": laplacian_sharpening,
        "high_pass_filter": high_pass_filter,
        "normalize_and_enhance": normalize_and_enhance,
        "multi_scale_sharpening": multi_scale_sharpening,
        "prepare_for_background_removal": prepare_for_background_removal,
        "edge_enhanced_preprocessing": edge_enhanced_preprocessing,
        "kmeans_segmentation": kmeans_segmentation,
        "adaptive_threshold_preprocessing": adaptive_threshold_preprocessing,
        "watershed_segmentation": watershed_segmentation,
        "focus_enhancement": focus_enhancement
    }
    
    # Define parameters for each method
    method_params = {
        "unsharp_masking": {"kernel_size": (5, 5), "sigma": 1.0, "amount": 1.5},
        "clahe": {"clip_limit": 2.5},
        "wiener_deconvolution": {"psf_size": 5, "noise_power": 0.01},
        "bilateral_filter": {"d": 9, "sigma_color": 75, "sigma_space": 75},
        "guided_filter": {"radius": 5, "eps": 0.1},
        "laplacian_sharpening": {"kernel_size": 3, "scale": 0.5},
        "high_pass_filter": {"sigma": 1.0},
        "normalize_and_enhance": {},
        "multi_scale_sharpening": {"scales": [3, 5, 9], "weights": [0.5, 0.3, 0.2]},
        "prepare_for_background_removal": {},
        "edge_enhanced_preprocessing": {"low_threshold": 10, "high_threshold": 60},
        "kmeans_segmentation": {"k": 3},
        "adaptive_threshold_preprocessing": {"block_size": 11, "c": 2},
        "watershed_segmentation": {},
        "focus_enhancement": {"kernel_size": 5, "sharp_amount": 2.5}
    }
    
    print("Processing images with different enhancement methods...")
    # Process each image with each method
    method_data = {}
    for method_name in methods:
        print(f"Processing {method_name}...")
        method_data[method_name] = []
        method_func = method_funcs.get(method_name)
        params = method_params.get(method_name, {}) if method_name != "original" else {}
        
        for i in range(1, 361):  # Process all 360 images
            filename = f"tool{i:03d}.jpg"
            image_path = os.path.join(base_tool_dir, filename)
            
            if not os.path.exists(image_path):
                continue
                
            processed = process_with_method(
                image_path, 
                median_bg, 
                method_func=method_func, 
                method_params=params
            )
            
            # Save processed image
            output_path = os.path.join(method_dirs[method_name], filename)
            cv2.imwrite(output_path, processed)
            
            # Calculate white pixel count
            pixel_count = analyze_white_pixels(processed)
            method_data[method_name].append((i, pixel_count))
    
    # Create a separate plot for each method
    print("Generating individual plots...")
    for method_name, data in method_data.items():
        if not data:
            continue
            
        degrees, counts = zip(*data)
        
        plt.figure(figsize=(12, 6))
        plt.plot(degrees, counts, label=method_name, linewidth=2)
        plt.title(f'{method_name} After Background Subtraction')
        plt.xlabel('Image Number')
        plt.ylabel('White Pixel Count')
        plt.ylim(4250, 4750)  # Set y-axis limits
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save the individual plot
        plot_path = os.path.join(plots_dir, f"{method_name}_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()
    
    # Also create one combined plot with all methods (for reference)
    plt.figure(figsize=(15, 8))
    for method_name, data in method_data.items():
        if not data:
            continue
            
        degrees, counts = zip(*data)
        plt.plot(degrees, counts, label=method_name)
    
    plt.title('Comparison of All Enhancement Methods')
    plt.xlabel('Image Number')
    plt.ylabel('White Pixel Count')
    plt.ylim(4250, 4750)  # Set y-axis limits
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(plots_dir, "all_methods_comparison_plot.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"Background subtraction and analysis completed. Results saved to {comparision_dir}")

if __name__ == "__main__":
    # Create visualizations and run analysis
    analyze_enhancement_methods() 