import os
import cv2
import numpy as np
from PIL import Image
from ..main import CONFIG
from .image_utils import display_images

def run_playground():

    print("--- Starting Multi-Channel Playground Script ---")

    ## --- 1. Load Image ---
    try:
        sample_image_path = 'image_to_signal/data/tool069gain10paperBG_blurred/0000.98_degrees.tiff'
        print(f"Loading sample image: {sample_image_path}")
        original_pil_image = Image.open(sample_image_path)
    except FileNotFoundError:
        print(f"Error: Cannot find the image.")
        return

    ## --- 2. Prepare Color Spaces ---
    rgb_image_np = np.array(original_pil_image.convert('RGB'))
    # Create BOTH LAB and HSV versions of the image
    lab_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2LAB)
    hsv_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)
    print("Created LAB and HSV versions of the image.")

    ## --- 3. Define All Thresholds (CONTROL PANEL) ---
    # -- LAB Thresholds (real-world scale) --
    L_threshold = 0      # Keep pixels with Lightness > 0
    a_threshold = 0     # Keep pixels with a <= -2
    b_threshold = -8      # Keep pixels with b < -8

    # -- HSV Thresholds (H: 0-179, S: 0-255, V: 0-255) --
    # H_threshold_min = int(220/2)  # Corresponds to 220 degrees
    # H_threshold_max = int(270/2)  # Corresponds to 270 degrees
    H_threshold = 70 // 2      # Hue > 220 degrees (0-179 scale)
    # S_threshold = 38       # Saturation > 15% (0.15 * 255)
    s_threshold_min = 15*(255/100)       # Saturation > 15%
    s_threshold_max = 70*(255/100)      # Saturation <= 100%
    # V_threshold = 0        # Value > 0%
    V_threshold_min = 45*(255/100)      # Value > 50%
    V_threshold_max = 55*(255/100)     # Value <= 52%

    ## --- 4. Isolate All 6 Channels ---
    L_channel = lab_image_np[:, :, 0]
    a_channel = lab_image_np[:, :, 1]
    b_channel = lab_image_np[:, :, 2]
    
    H_channel = hsv_image_np[:, :, 0]
    S_channel = hsv_image_np[:, :, 1]
    V_channel = hsv_image_np[:, :, 2]

    ## --- 5. Create a Boolean Mask for Each Condition ---
    # Note: `~` is the NOT operator, inverting a mask.
    # LAB Masks
    L_mask = L_channel > L_threshold
    a_mask = a_channel <= (a_threshold + 128)
    b_mask = b_channel < (b_threshold + 128)
    
    # HSV Masks
    # H_mask = cv2.inRange(H_channel, H_threshold_min, H_threshold_max).astype(bool)
    H_mask = H_channel < H_threshold
    # S_mask = S_channel > S_threshold
    S_mask = (S_channel >= s_threshold_min) & (S_channel <= s_threshold_max)
    V_mask = (V_channel > V_threshold_min) & (V_channel <= V_threshold_max)

    ## --- 6. Combine Masks to Create the Final Mask ---
    # Here is where you play! Combine the 6 masks using & (AND) and | (OR).
    # Parentheses control the order of operations.

    # Example 1: A strict mask requiring all LAB conditions to be true.
    # final_boolean_mask = (L_mask) & (a_mask) & (b_mask)
    # condition_text = "L & a & b"

    # Example 2: Your "two-step" idea. This finds pixels that meet the 'a' AND 'H' conditions.
    # This is the most efficient way to implement "if a_mask is true, then check H_mask".
    # final_boolean_mask = (a_mask) & (H_mask)
    # condition_text = "a AND H"

    # Example 3: A more complex rule.
    # (Must meet 'a' condition) AND (Must meet EITHER 'H' OR 'S' condition)
    final_boolean_mask =   (b_mask)  | (a_mask) | ~(V_mask)
    condition_text = "b and a or not V"

    ## --- 7. Generate and Display Final Result ---
    final_mask_visual = final_boolean_mask.astype(np.uint8) * 255
    mask_pil = Image.fromarray(final_mask_visual)
    
    print(f"Created final mask using logic: '{condition_text}'.")

    display_images([
        (original_pil_image, 'Original Blurred Image'),
        (mask_pil, f'({condition_text})')
    ])

    print("\n--- Playground Script Finished ---")

if __name__ == "__main__":
    run_playground()