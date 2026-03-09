import argparse
import cv2
import numpy as np
import os
import glob

# ==========================================
# 1. SETUP FOLDER PATHS
# ==========================================
DEFAULT_TOOL_ID = "tool012"
DEFAULT_BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA"


def build_paths(base_dir, tool_id):
    input_masks_dir = os.path.join(base_dir, "DATA", "masks", f"{tool_id}_final_masks")
    output_root = os.path.join(base_dir, "Tool_Condition_Monitoring", "two_edge_case", "swept_volume_rotation", "output")

    return {
        "input_masks_dir": input_masks_dir,
        "out_aligned_masks": os.path.join(output_root, f"{tool_id}_aligned_masks"),
        "out_debug_lines": os.path.join(output_root, f"{tool_id}_debug_lines_angles"),
        "out_debug_rulers": os.path.join(output_root, f"{tool_id}_debug_rulers_fixed"),
    }


def ensure_output_dirs(paths):
    for folder in [
        paths["out_aligned_masks"],
        paths["out_debug_lines"],
        paths["out_debug_rulers"],
    ]:
        os.makedirs(folder, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def find_tiff_files(input_dir):
    search_pattern = os.path.join(input_dir, "*.[tT][iI][fF]*")
    return sorted(glob.glob(search_pattern))

def to_binary_mask(mask_img):
    if mask_img.ndim == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

# ==========================================
# 3. BUILD THE MASTER MASK
# ==========================================
def build_master_mask(file_list):
    print(f"Building 360-degree Master Mask from {len(file_list)} frames...")
    
    # Read first image to get dimensions
    first_img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
    master_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    
    for file_path in file_list:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            binary = to_binary_mask(img)
            # Bitwise OR combines all white pixels across all frames
            master_mask = cv2.bitwise_or(master_mask, binary)
            
    return master_mask

# ==========================================
# 4. CALCULATE TILT FROM MASTER MASK
# ==========================================
def calculate_master_tilt(master_mask, output_lines_dir):
    h, w = master_mask.shape
    ys, xs = np.where(master_mask == 255)
    
    if len(ys) == 0:
        return 0.0
        
    top_y = np.min(ys)
    bottom_y = np.max(ys)
    tool_length = bottom_y - top_y
    
    # --- ISOLATE THE SAFE ZONE ---
    # Start 5% down (avoid top edge noise). 
    # Stop at 60% down (safely avoids the bottom 40% where broken tips exist).
    roi_start_y = int(top_y + (tool_length * 0.05))
    roi_end_y = int(top_y + (tool_length * 0.60))
    
    left_points = []
    right_points = []
    
    for y in range(roi_start_y, roi_end_y):
        row = master_mask[y, :]
        white_pixels = np.where(row == 255)[0]
        if len(white_pixels) > 0:
            left_points.append((white_pixels[0], y))
            right_points.append((white_pixels[-1], y))
            
    left_pts = np.array(left_points)
    right_pts = np.array(right_points)
    
    # Fit lines
    m_left, c_left = np.polyfit(left_pts[:, 1], left_pts[:, 0], 1)
    m_right, c_right = np.polyfit(right_pts[:, 1], right_pts[:, 0], 1)
    
    m_center = (m_left + m_right) / 2
    tilt_angle_deg = np.degrees(np.arctan(m_center))
    
    # --- DRAW VISUALIZATION ---
    vis = cv2.cvtColor(master_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (0, roi_start_y), (w, roi_end_y), (0, 255, 255), 2)
    
    # Left Line (Red)
    pt1_l = (int(m_left * roi_start_y + c_left), roi_start_y)
    pt2_l = (int(m_left * roi_end_y + c_left), roi_end_y)
    cv2.line(vis, pt1_l, pt2_l, (0, 0, 255), 4)
    
    # Right Line (Green)
    pt1_r = (int(m_right * roi_start_y + c_right), roi_start_y)
    pt2_r = (int(m_right * roi_end_y + c_right), roi_end_y)
    cv2.line(vis, pt1_r, pt2_r, (0, 255, 0), 4)
    
    cv2.putText(vis, f"Calculated Rotation Axis Tilt: {tilt_angle_deg:.3f} deg", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 4)
    
    out_path = os.path.join(output_lines_dir, "DEBUG_MASTER_MASK_CALIBRATION.png")
    cv2.imwrite(out_path, vis)
    
    return tilt_angle_deg

# ==========================================
# 5. DRAW VERTICAL RULERS
# ==========================================
def draw_vertical_rulers(mask_binary, original_filename, output_rulers_dir):
    h, w = mask_binary.shape
    ys, xs = np.where(mask_binary == 255)
    vis = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
    
    if len(ys) > 0:
        top_y = np.min(ys)
        tool_length = np.max(ys) - top_y
        
        # Place rulers based on upper section
        roi_start_y = int(top_y + (tool_length * 0.10))
        roi_end_y = int(top_y + (tool_length * 0.30))
        
        shank_xs = []
        for y in range(roi_start_y, roi_end_y):
            row = mask_binary[y, :]
            wp = np.where(row == 255)[0]
            if len(wp) > 0:
                shank_xs.extend([wp[0], wp[-1]])
                
        if shank_xs:
            leftmost_x = np.min(shank_xs)
            rightmost_x = np.max(shank_xs)
            cv2.line(vis, (leftmost_x, 0), (leftmost_x, h), (255, 0, 0), 3)
            cv2.line(vis, (rightmost_x, 0), (rightmost_x, h), (255, 0, 0), 3)
            
    out_path = os.path.join(output_rulers_dir, original_filename)
    cv2.imwrite(out_path, vis)

# ==========================================
# 6. MAIN EXECUTION LOOP
# ==========================================
def process_all_frames(tool_id=DEFAULT_TOOL_ID, base_dir=DEFAULT_BASE_DIR):
    paths = build_paths(base_dir, tool_id)
    ensure_output_dirs(paths)

    file_list = find_tiff_files(paths["input_masks_dir"])
    if not file_list:
        print(f"No files found in {paths['input_masks_dir']}")
        return

    # 1. Build Master Mask from all frames
    master_mask = build_master_mask(file_list)
    
    # 2. Calibrate using the Master Mask
    tilt_angle = calculate_master_tilt(master_mask, paths["out_debug_lines"])
    rotation_angle = -tilt_angle 
    
    print(f"Calibration Complete! True axis tilt: {tilt_angle:.3f} degrees.")
    print(f"Applying counter-rotation of {rotation_angle:.3f} degrees to all {len(file_list)} frames...\n")

    # 3. Get Rotation Matrix
    h, w = master_mask.shape
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # 4. Apply to all frames
    for idx, file_path in enumerate(file_list, start=1):
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue
            
        b_mask = to_binary_mask(img)
        rotated_mask = cv2.warpAffine(b_mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(paths["out_aligned_masks"], filename), rotated_mask)
        draw_vertical_rulers(rotated_mask, filename, paths["out_debug_rulers"])
        
        if idx % 50 == 0 or idx == len(file_list):
            print(f"  Processed [{idx}/{len(file_list)}] -> {filename}")

    print(f"\nAll done for {tool_id}! Master Mask calibration image saved in the debug_lines folder.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate perspective tilt for one tool and export aligned masks."
    )
    parser.add_argument(
        "--tool-id",
        default=DEFAULT_TOOL_ID,
        help="Tool identifier, for example: tool012",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Root CCD_DATA directory containing DATA/ and Tool_Condition_Monitoring/",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all_frames(tool_id=args.tool_id, base_dir=args.base_dir)