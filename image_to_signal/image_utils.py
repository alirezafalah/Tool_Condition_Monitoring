from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

def display_images(images_with_titles):
    """
    Displays one or more images side-by-side.

    Args:
        images_with_titles (list): A list of tuples, where each tuple is
                                   (image_object_or_path, title_string).
    """
    # Ensure the input is a list, even if only one image is passed
    if not isinstance(images_with_titles, list):
        # This allows calling the function with a single tuple without wrapping it in a list
        images_with_titles = [images_with_titles]
    
    num_images = len(images_with_titles)
    # Create a figure with a row of subplots for each image
    fig, axes = plt.subplots(1, num_images, figsize=(6 * num_images, 6), sharex=True, sharey=True)

    # If there's only one image, 'axes' is not a list, so we make it one
    if num_images == 1:
        axes = [axes]

    try:
        # Loop through the axes and the list of (image, title) tuples
        for ax, (img_data, title) in zip(axes, images_with_titles):
            img = None
            # Check if we were given a path or an image object
            if isinstance(img_data, str):
                img = Image.open(img_data)
            elif isinstance(img_data, Image.Image):
                img = img_data
            else:
                raise TypeError("Image data must be a file path or a PIL Image object.")

            ax.imshow(img)
            ax.set_title(title, fontsize=14)
            ax.axis('off') # Hide the x and y axes

        plt.tight_layout() # Adjusts spacing between plots
        plt.show()

    except Exception as e:
        print(f"An error occurred while displaying images: {e}")


def convert_tiff_to_jpeg(tiff_path, quality=95):
    """
    Convert a TIFF image to JPEG format with the same filename.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        quality (int): JPEG quality (1-100, default 95)
    """
    try:
        with Image.open(tiff_path) as img:
            # Convert to RGB if necessary (JPEG doesn't support transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create JPEG path with same name but .jpg extension
            jpeg_path = tiff_path.rsplit('.', 1)[0] + '.jpg'
            
            img.save(jpeg_path, 'JPEG', quality=quality)
            print(f"Successfully converted '{tiff_path}' to '{jpeg_path}'")
    except FileNotFoundError:
        print(f"Error: TIFF file '{tiff_path}' not found.")
    except Exception as e:
        print(f"Error converting image: {e}")



