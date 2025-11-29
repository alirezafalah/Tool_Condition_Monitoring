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
    parser.add_argument('--block-plot', action='store_true', help='Keep plot window open until closed (blocking)')
    parser.add_argument('--write-json', action='store_true', help='Write results JSON (for GUI embedding)')
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
        # Interactive matplotlib window (smaller footprint ~640x300)
        fig = plt.figure(figsize=(6.4, 3.0))
        nums = [r['frame_number'] for r in results]

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1], figure=fig)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_ref = fig.add_subplot(gs[0, 1])
        ax_best = fig.add_subplot(gs[1, 1])

        ax_main.plot(nums, [r['white_i'] for r in results], 'b-', label='Frame i', linewidth=1.2)
        ax_main.plot([n + 1 for n in nums], [r['white_j'] for r in results], 'c--', label='Frame i+1', linewidth=1.0)
        ax_main.axhline(first_white, color='green', linestyle='--', linewidth=0.9, label=f'Ref1 {first_white:,}')
        ax_main.axhline(second_white, color='lime', linestyle='--', linewidth=0.9, label=f'Ref2 {second_white:,}')
        ax_main.axvline(best['frame_number'], color='red', linewidth=1.3, label=f"Best {best['frame_number']}")
        ax_main.set_xlabel('Frame Number', fontsize=9)
        ax_main.set_ylabel('White Pixels', fontsize=9)
        ax_main.set_title(f'360° Similarity (tool {tool_id})', fontsize=10)
        ax_main.legend(fontsize=7, ncol=2)
        ax_main.grid(alpha=0.25)

        ref_combo = np.hstack([first_mask, second_mask])
        ax_ref.imshow(ref_combo, cmap='gray')
        ax_ref.axis('off')
        ax_ref.set_title(f'Ref 1 & 2\n{first_white:,} | {second_white:,}', fontsize=8)

        best_i = np.array(Image.open(os.path.join(masks_dir, image_files[best['frame_idx']])))
        best_j = np.array(Image.open(os.path.join(masks_dir, image_files[best['frame_idx'] + 1])))
        best_combo = np.hstack([best_i, best_j])
        ax_best.imshow(best_combo, cmap='gray')
        ax_best.axis('off')
        ax_best.set_title(f'Best {best["frame_number"]} & {best["frame_number"]+1}\n{best["white_i"]:,} | {best["white_j"]:,}', fontsize=8)

        fig.tight_layout()
        if args.block_plot:
            # Blocking show keeps window until user closes it (prevents auto-close)
            plt.show()
        else:
            # Non-blocking show will close immediately when process exits
            # (Use --block-plot when launched via subprocess to keep it visible)
            try:
                plt.show(block=False)
                plt.pause(0.5)  # brief event pump; window will still vanish on process exit
            except Exception:
                pass

    remove = total_frames - best['frame_number']
    print("\nAction:"); sys.stdout.flush()
    if remove > 0:
        print(f"   Remove the last {remove} frame(s) to keep {best['frame_number']} = 360 degrees"); sys.stdout.flush()
    else:
        print(f"   Dataset already aligned to 360 degrees with {best['frame_number']} frames"); sys.stdout.flush()

    if args.write_json:
        import json
        out_dir = os.path.join(DATA_ROOT, '1d_profiles')
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f'find360_{tool_id}.json')
        payload = {
            'tool_id': tool_id,
            'first_white': int(first_white),
            'second_white': int(second_white),
            'best_frame_number': int(best['frame_number']),
            'best_white_i': int(best['white_i']),
            'best_white_j': int(best['white_j']),
            'frame_numbers': [int(r['frame_number']) for r in results],
            'white_i_series': [int(r['white_i']) for r in results],
            'white_j_series': [int(r['white_j']) for r in results],
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            print(f"Results JSON written: {json_path}"); sys.stdout.flush()
        except Exception as e:
            print(f"Failed to write JSON: {e}"); sys.stdout.flush()

if __name__ == '__main__':
    main()
