"""Rename mask and blurred images to include rotation angle after determining frames for 360°.

Run:
  python -m image_to_signal.rename_by_angle --tool tool002 --frames360 363
  python -m image_to_signal.rename_by_angle --tool tool002 --frames360 363 --folder blurred

This will rename files in DATA/masks/<tool_id>_final_masks/ or DATA/blurred/<tool_id>_blurred/
to <angle>_degrees.tiff where angle advances in steps of 360/frames360.
"""
import os
import sys
import argparse
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATA"))

def rename(tool_id: str, frames360: int, folder_type: str = 'masks', force: bool = False):
    if folder_type == 'masks':
        target_dir = os.path.join(DATA_ROOT, 'masks', f'{tool_id}_final_masks')
    elif folder_type == 'blurred':
        target_dir = os.path.join(DATA_ROOT, 'blurred', f'{tool_id}_blurred')
    else:
        print(f"Invalid folder type: {folder_type}. Use 'masks' or 'blurred'.")
        sys.stdout.flush()
        return
    
    if not os.path.isdir(target_dir):
        print(f"Directory not found: {target_dir}")
        sys.stdout.flush()
        return

    files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.tiff', '.tif'))])
    if not files:
        print(f"No image files found in {folder_type} directory.")
        sys.stdout.flush()
        return

    # Detect if already renamed
    already_renamed = any('_degrees.tiff' in f for f in files)
    if already_renamed and not force:
        print("ALREADY_RENAMED")  # Special marker for GUI to detect
        sys.stdout.flush()
        return

    angle_step = 360.0 / frames360
    print(f"Renaming {len(files)} {folder_type} files using angle step {angle_step:.6f} (frames360={frames360})...")
    sys.stdout.flush()
    for i, fname in enumerate(tqdm(files, desc=f"Renaming {folder_type}")):
        angle = i * angle_step
        new_name = f"{angle:07.2f}_degrees.tiff"
        old_path = os.path.join(target_dir, fname)
        new_path = os.path.join(target_dir, new_name)
        try:
            os.rename(old_path, new_path)
        except OSError as e:
            print(f"Error renaming {fname}: {e}")
            sys.stdout.flush()
    print("Done.")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Rename mask or blurred images with angle after 360 detection.")
    parser.add_argument('--tool', required=True, help='Tool ID, e.g. tool002')
    parser.add_argument('--frames360', required=True, type=int, help='Number of frames representing 360° (from find360)')
    parser.add_argument('--folder', default='masks', choices=['masks', 'blurred'], help='Folder type to rename (default: masks)')
    parser.add_argument('--force', action='store_true', help='Force rename even if already renamed')
    args = parser.parse_args()
    rename(args.tool, args.frames360, args.folder, args.force)

if __name__ == '__main__':
    main()