import os
import glob
import csv
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION (kept consistent with run_all_tools_analysis.py selection logic)
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
TOOLS_METADATA_PATH = os.path.join(BASE_DIR, "tools_metadata.csv")

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "threshold_analysis",
    "master_mask_perspective",
)
OUT_MASTER_MASKS = os.path.join(OUTPUT_ROOT, "master_masks")
OUT_DEBUG_LINES = os.path.join(OUTPUT_ROOT, "debug_master_mask_calibration")
OUT_TOOL_METADATA = os.path.join(OUTPUT_ROOT, "tool_metadata")

# Keep same skip list behavior as run_all_tools_analysis.py
SKIP_TOOLS = ["tool016", "tool069"]


# ============================================================================
# METADATA + TOOL DISCOVERY (same logic as run_all_tools_analysis.py)
# ============================================================================
def load_tools_metadata():
    metadata = {}
    if os.path.exists(TOOLS_METADATA_PATH):
        with open(TOOLS_METADATA_PATH, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                metadata[row["tool_id"]] = row
    return metadata


def get_mask_folder(tool_id):
    patterns = [
        f"{tool_id}_final_masks",
        f"{tool_id}gain10paperBG_final_masks",
        f"{tool_id}gain10_final_masks",
    ]

    for pattern in patterns:
        folder = os.path.join(MASKS_DIR, pattern)
        if os.path.exists(folder):
            return folder
    return None


def get_mask_files(mask_folder):
    pattern = os.path.join(mask_folder, "*.tiff")
    files = glob.glob(pattern)

    if not files:
        pattern = os.path.join(mask_folder, "*.tif")
        files = glob.glob(pattern)

    if not files:
        return []

    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
        name = basename.replace(".tiff", "").replace(".tif", "")

        import re

        match = re.match(r"^(\d+\.?\d*)", name)
        if match:
            return float(match.group(1))

        parts = name.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return float(part)
            try:
                return float(part)
            except ValueError:
                continue
        return 0.0

    return sorted(files, key=extract_frame_num)


def get_two_edge_tools(tools_metadata):
    two_edge_tools = []
    for tool_id, meta in tools_metadata.items():
        if tool_id in SKIP_TOOLS:
            continue

        edges = meta.get("edges", "")
        try:
            if int(edges) == 2:
                mask_folder = get_mask_folder(tool_id)
                if mask_folder is not None:
                    mask_files = get_mask_files(mask_folder)
                    if len(mask_files) > 0:
                        two_edge_tools.append(tool_id)
        except Exception:
            pass

    return sorted(two_edge_tools)


# ============================================================================
# MASTER MASK + TILT CALIBRATION
# ============================================================================
def to_binary_mask(mask_img):
    if mask_img.ndim == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def build_master_mask(file_list):
    first_img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        return None

    master_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)

    for file_path in file_list:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        binary = to_binary_mask(img)
        master_mask = cv2.bitwise_or(master_mask, binary)

    return master_mask


def calculate_master_tilt(master_mask, debug_out_path):
    h, w = master_mask.shape
    ys, _ = np.where(master_mask == 255)

    if len(ys) == 0:
        return {
            "tilt_angle_deg": 0.0,
            "rotation_angle_deg": -0.0,
            "roi_start_y": None,
            "roi_end_y": None,
            "status": "empty_master_mask",
        }

    top_y = int(np.min(ys))
    bottom_y = int(np.max(ys))
    tool_length = bottom_y - top_y

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

    if len(left_points) < 2 or len(right_points) < 2:
        vis = cv2.cvtColor(master_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (0, roi_start_y), (w, roi_end_y), (0, 255, 255), 2)
        cv2.putText(
            vis,
            "Insufficient points for line fit",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )
        cv2.imwrite(debug_out_path, vis)
        return {
            "tilt_angle_deg": 0.0,
            "rotation_angle_deg": -0.0,
            "roi_start_y": roi_start_y,
            "roi_end_y": roi_end_y,
            "status": "insufficient_points_for_fit",
        }

    left_pts = np.array(left_points)
    right_pts = np.array(right_points)

    m_left, c_left = np.polyfit(left_pts[:, 1], left_pts[:, 0], 1)
    m_right, c_right = np.polyfit(right_pts[:, 1], right_pts[:, 0], 1)

    m_center = (m_left + m_right) / 2
    tilt_angle_deg = float(np.degrees(np.arctan(m_center)))
    rotation_angle_deg = -tilt_angle_deg

    vis = cv2.cvtColor(master_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (0, roi_start_y), (w, roi_end_y), (0, 255, 255), 2)

    pt1_l = (int(m_left * roi_start_y + c_left), roi_start_y)
    pt2_l = (int(m_left * roi_end_y + c_left), roi_end_y)
    cv2.line(vis, pt1_l, pt2_l, (0, 0, 255), 4)

    pt1_r = (int(m_right * roi_start_y + c_right), roi_start_y)
    pt2_r = (int(m_right * roi_end_y + c_right), roi_end_y)
    cv2.line(vis, pt1_r, pt2_r, (0, 255, 0), 4)

    cv2.putText(
        vis,
        f"Tilt: {tilt_angle_deg:.3f} deg | Counter-Rotation: {rotation_angle_deg:.3f} deg",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 0),
        3,
    )

    cv2.imwrite(debug_out_path, vis)

    return {
        "tilt_angle_deg": tilt_angle_deg,
        "rotation_angle_deg": rotation_angle_deg,
        "roi_start_y": roi_start_y,
        "roi_end_y": roi_end_y,
        "m_left": float(m_left),
        "m_right": float(m_right),
        "status": "ok",
    }


def process_tool(tool_id, meta):
    mask_folder = get_mask_folder(tool_id)
    if mask_folder is None:
        return None

    mask_files = get_mask_files(mask_folder)
    if not mask_files:
        return None

    master_mask = build_master_mask(mask_files)
    if master_mask is None:
        return None

    master_mask_path = os.path.join(OUT_MASTER_MASKS, f"{tool_id}_MASTER_MASK.tiff")
    cv2.imwrite(master_mask_path, master_mask)

    debug_path = os.path.join(OUT_DEBUG_LINES, f"{tool_id}_DEBUG_MASTER_MASK_CALIBRATION.png")
    tilt_info = calculate_master_tilt(master_mask, debug_path)

    record = {
        "tool_id": tool_id,
        "tool_type": meta.get("type", ""),
        "condition": meta.get("condition", ""),
        "edges": meta.get("edges", ""),
        "mask_folder": mask_folder,
        "num_frames": len(mask_files),
        "master_mask_path": master_mask_path,
        "debug_calibration_path": debug_path,
        "tilt_angle_deg": tilt_info.get("tilt_angle_deg", 0.0),
        "rotation_angle_deg": tilt_info.get("rotation_angle_deg", 0.0),
        "roi_start_y": tilt_info.get("roi_start_y"),
        "roi_end_y": tilt_info.get("roi_end_y"),
        "status": tilt_info.get("status", "ok"),
    }

    per_tool_meta_path = os.path.join(OUT_TOOL_METADATA, f"{tool_id}_master_mask_metadata.json")
    with open(per_tool_meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                **record,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            file,
            indent=2,
        )

    record["tool_metadata_json"] = per_tool_meta_path
    return record


def main():
    os.makedirs(OUT_MASTER_MASKS, exist_ok=True)
    os.makedirs(OUT_DEBUG_LINES, exist_ok=True)
    os.makedirs(OUT_TOOL_METADATA, exist_ok=True)

    print("=" * 80)
    print("BUILD MASTER MASKS FOR ALL 2-EDGE TOOLS")
    print("=" * 80)
    print(f"Tools metadata: {TOOLS_METADATA_PATH}")
    print(f"Masks dir: {MASKS_DIR}")
    print(f"Skip tools: {SKIP_TOOLS}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 80)

    tools_metadata = load_tools_metadata()
    two_edge_tools = get_two_edge_tools(tools_metadata)

    print(f"Found {len(two_edge_tools)} candidate 2-edge tools with mask images.\n")

    rows = []
    for idx, tool_id in enumerate(two_edge_tools, start=1):
        meta = tools_metadata.get(tool_id, {})
        print(f"[{idx}/{len(two_edge_tools)}] Processing {tool_id}...", end=" ")

        result = process_tool(tool_id, meta)
        if result is None:
            print("FAILED")
            continue

        print(
            f"OK | frames={result['num_frames']} | "
            f"tilt={result['tilt_angle_deg']:.3f} deg | "
            f"status={result['status']}"
        )
        rows.append(result)

    if not rows:
        print("\nNo tools were processed successfully.")
        return

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(OUTPUT_ROOT, "master_mask_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    run_metadata = {
        "method": "Master Mask Perspective Calibration",
        "description": "Builds OR-combined master mask and computes axis tilt per 2-edge tool",
        "selection_logic": {
            "edges_equals": 2,
            "skip_tools": SKIP_TOOLS,
            "mask_folder_patterns": [
                "{tool_id}_final_masks",
                "{tool_id}gain10paperBG_final_masks",
                "{tool_id}gain10_final_masks",
            ],
            "requires_non_empty_mask_folder": True,
        },
        "tools_found": len(two_edge_tools),
        "tools_processed_successfully": len(rows),
        "output_root": OUTPUT_ROOT,
        "summary_csv": summary_csv,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    run_meta_path = os.path.join(OUTPUT_ROOT, "run_metadata.json")
    with open(run_meta_path, "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Processed tools: {len(rows)} / {len(two_edge_tools)}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Run metadata: {run_meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()