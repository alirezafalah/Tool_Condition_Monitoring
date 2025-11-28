"""Sequential similarity based 360° frame count detector.

Usage:
    python -m image_to_signal.find360 --tool tool002

Optional:
    python -m image_to_signal.find360 --tool tool002 --test-window 25 --expected-366 366
"""
import os
import argparse
import traceback
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATA"))


def count_white_pixels(mask_np):
    return np.sum(mask_np == 255)


def calculate_pixel_difference(mask1_np, mask2_np):
    c1 = count_white_pixels(mask1_np)
    c2 = count_white_pixels(mask2_np)
    abs_diff = abs(c1 - c2)
    avg = (c1 + c2) / 2
    rel_diff = abs_diff / avg if avg > 0 else float('inf')
    return abs_diff, rel_diff, c1, c2


def main():
    try:
        _run()
    except Exception:
        print("\n[find360] Unhandled exception:")
        traceback.print_exc()
        raise

def _run():
    parser = argparse.ArgumentParser(description="Find 360° frame count by sequential pixel similarity.")
    parser.add_argument('--tool', default='tool002', help='Tool ID, e.g. tool002')
    parser.add_argument('--test-window', type=int, default=25, help='Number of frames to inspect near expected wrap')
    parser.add_argument('--expected-366', type=int, default=366, help='Angle basis (default 366° recorded)')
    parser.add_argument('--no-plot', action='store_true', help='Suppress matplotlib visualization (GUI mode)')
    args = parser.parse_args()

    tool_id = args.tool
    masks_dir = os.path.join(DATA_ROOT, 'masks', f'{tool_id}_final_masks')

    print("=" * 70); sys.stdout.flush()
    print("FIND 360° BY SEQUENTIAL FRAME SIMILARITY"); sys.stdout.flush()
    print("=" * 70); sys.stdout.flush()
    print(f"Tool ID: {tool_id}"); sys.stdout.flush()
    print(f"Masks directory: {masks_dir}"); sys.stdout.flush()

    try:
        image_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No masks found in '{masks_dir}'.")
            return
    except FileNotFoundError:
        print("Error: Mask directory not found. Check tool ID.")
        return

    total_frames = len(image_files)
    print(f"\nTotal available frames: {total_frames}"); sys.stdout.flush()

    if len(image_files) < 2:
        print("Need at least 2 frames for sequential comparison.")
        return

    first_mask = np.array(Image.open(os.path.join(masks_dir, image_files[0])))
    second_mask = np.array(Image.open(os.path.join(masks_dir, image_files[1])))
    first_white = count_white_pixels(first_mask)
    second_white = count_white_pixels(second_mask)
    print(f"First frame white pixels: {first_white:,}"); sys.stdout.flush()
    print(f"Second frame white pixels: {second_white:,}"); sys.stdout.flush()

    expected_360 = int((360.0 / args.expected_366) * total_frames)
    start_idx = max(expected_360 - 15, total_frames - args.test_window)
    end_idx = min(total_frames, expected_360 + 15)
    print(f"\nExpected 360° frame: ~{expected_360}"); sys.stdout.flush()
    print(f"Testing frames {start_idx} to {end_idx-2} (checking pairs)..."); sys.stdout.flush()

    results = []
    print("\nComparing frame PAIRS to first two frames..."); sys.stdout.flush()
    for idx in tqdm(range(start_idx, end_idx - 1), desc="Testing frame pairs"):
        frame_i = np.array(Image.open(os.path.join(masks_dir, image_files[idx])))
        frame_j = np.array(Image.open(os.path.join(masks_dir, image_files[idx + 1])))
        abs1, rel1, _, c_i = calculate_pixel_difference(first_mask, frame_i)
        abs2, rel2, _, c_j = calculate_pixel_difference(second_mask, frame_j)
        score = (rel1 + rel2) / 2
        results.append({
            'frame_idx': idx,
            'frame_number': idx + 1,
            'white_i': c_i,
            'white_j': c_j,
            'rel1': rel1,
            'rel2': rel2,
            'score': score,
            'score_pct': score * 100
        })

    if not results:
        print("No results generated in test window.")
        return

    results_sorted = sorted(results, key=lambda r: r['score'])
    print("\n" + "=" * 70); sys.stdout.flush()
    print("TOP 10 MATCHES (sorted by combined similarity):"); sys.stdout.flush()
    print("=" * 70); sys.stdout.flush()
    print(f"{'Frame Pair':>11} | {'Frame i':>13} | {'Frame i+1':>13} | {'Score %':>10}"); sys.stdout.flush()
    print("-" * 70); sys.stdout.flush()
    print(f"{'REFERENCE':>11} | {first_white:>13,} | {second_white:>13,} | {'-':>10}"); sys.stdout.flush()
    print("-" * 70); sys.stdout.flush()
    for r in results_sorted[:10]:
        pair = f"{r['frame_number']}-{r['frame_number']+1}"
        print(f"{pair:>11} | {r['white_i']:>13,} | {r['white_j']:>13,} | {r['score_pct']:>10.2f}"); sys.stdout.flush()

    best = results_sorted[0]
    print("\n" + "=" * 70); sys.stdout.flush()
    print(f"BEST MATCH: Frames #{best['frame_number']} and #{best['frame_number']+1}"); sys.stdout.flush()
    print(f"   Frame {best['frame_number']}: {best['white_i']:,} (ref: {first_white:,})"); sys.stdout.flush()
    print(f"   Frame {best['frame_number']+1}: {best['white_j']:,} (ref: {second_white:,})"); sys.stdout.flush()
    print(f"   Diff 1st pair: {best['rel1']*100:.2f}%, Diff 2nd pair: {best['rel2']*100:.2f}%"); sys.stdout.flush()
    print(f"   Combined score: {best['score_pct']:.2f}%"); sys.stdout.flush()
    print(f"   Use {best['frame_number']} frames for one complete 360 degree rotation"); sys.stdout.flush()
    print("=" * 70); sys.stdout.flush()

    if not args.no_plot:
        fig = plt.figure(figsize=(16, 8))
        fig.canvas.manager.set_window_title(f'360 Degree Detection - Tool {tool_id}')
        nums = [r['frame_number'] for r in results]
        
        # Left: Main graph - white pixel counts (takes 2/3 of width)
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(nums, [r['white_i'] for r in results], 'b-', label='Frame i', linewidth=2)
        ax1.plot([n+1 for n in nums], [r['white_j'] for r in results], 'c--', label='Frame i+1', linewidth=2)
        ax1.axhline(first_white, color='green', linestyle='--', linewidth=1.5, label=f'Ref 1 ({first_white:,})')
        ax1.axhline(second_white, color='lime', linestyle='--', linewidth=1.5, label=f'Ref 2 ({second_white:,})')
        ax1.axvline(best['frame_number'], color='red', linewidth=2.5, label=f"Best: {best['frame_number']}")
        ax1.set_xlabel('Frame Number', fontsize=11)
        ax1.set_ylabel('White Pixel Count', fontsize=11)
        ax1.set_title('White Pixel Counts - Sequential Pairs', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)

        # Right side: Two image comparisons stacked vertically
        ax2 = plt.subplot(2, 2, 2)
        ref_combo = np.hstack([first_mask, second_mask])
        ax2.imshow(ref_combo, cmap='gray')
        ax2.axis('off')
        ax2.set_title(f'Reference: Frames 1 & 2\n{first_white:,} | {second_white:,} pixels', fontsize=10)

        ax3 = plt.subplot(2, 2, 4)
        best_i = np.array(Image.open(os.path.join(masks_dir, image_files[best['frame_idx']])))
        best_j = np.array(Image.open(os.path.join(masks_dir, image_files[best['frame_idx']+1])))
        best_combo = np.hstack([best_i, best_j])
        ax3.imshow(best_combo, cmap='gray')
        ax3.axis('off')
        ax3.set_title(f'Best Match: Frames {best["frame_number"]} & {best["frame_number"]+1}\n{best["white_i"]:,} | {best["white_j"]:,} pixels', fontsize=10)

        plt.suptitle(f'360 Degree Detection - Tool {tool_id}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    remove = total_frames - best['frame_number']
    print("\nAction:"); sys.stdout.flush()
    if remove > 0:
        print(f"   Remove the last {remove} frame(s) to keep {best['frame_number']} = 360 degrees"); sys.stdout.flush()
    else:
        print(f"   Dataset already aligned to 360 degrees with {best['frame_number']} frames"); sys.stdout.flush()

if __name__ == '__main__':
    main()
