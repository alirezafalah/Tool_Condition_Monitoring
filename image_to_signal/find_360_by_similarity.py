"""
Find 360° rotation by comparing white pixel counts.
Finds which frame is most similar to the first frame by counting white pixels.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Calculate paths
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATA"))

# Tool ID (can be changed)
TOOL_ID = 'tool002'
TEST_MASKS_DIR = os.path.join(DATA_ROOT, 'masks', f'{TOOL_ID}_final_masks')


def count_white_pixels(mask_np):
    """Count total white pixels in mask."""
    return np.sum(mask_np == 255)


def calculate_pixel_difference(mask1_np, mask2_np):
    """Calculate absolute and relative difference in white pixel counts."""
    count1 = count_white_pixels(mask1_np)
    count2 = count_white_pixels(mask2_np)
    abs_diff = abs(count1 - count2)
    
    # Normalize difference
    avg_count = (count1 + count2) / 2
    rel_diff = abs_diff / avg_count if avg_count > 0 else float('inf')
    
    return abs_diff, rel_diff, count1, count2


def main():
    """Find which frame matches the first frame best."""
    print("=" * 70)
    print("FIND 360° BY WHITE PIXEL SIMILARITY")
    print("=" * 70)
    print(f"Tool ID: {TOOL_ID}")
    print(f"Masks directory: {TEST_MASKS_DIR}")
    
    # Load all masks
    try:
        image_files = sorted([f for f in os.listdir(TEST_MASKS_DIR) if f.endswith(('.tiff', '.tif'))])
        if not image_files:
            print(f"No masks found in '{TEST_MASKS_DIR}'.")
            return
    except FileNotFoundError:
        print(f"Error: Mask directory not found. Check TOOL_ID setting.")
        return
    
    total_frames = len(image_files)
    print(f"\nTotal available frames: {total_frames}")
    
    # Load first frame
    first_path = os.path.join(TEST_MASKS_DIR, image_files[0])
    first_mask = np.array(Image.open(first_path))
    first_white_count = count_white_pixels(first_mask)
    
    print(f"First frame white pixels: {first_white_count:,}")
    
    # Expected range for 360° (based on 12.2s @ 5 RPM = 366°)
    expected_360 = int((360.0 / 366.0) * total_frames)
    
    # Test last N frames
    N_FRAMES_TO_TEST = 25
    start_idx = max(expected_360 - 15, total_frames - N_FRAMES_TO_TEST)
    end_idx = min(total_frames, expected_360 + 15)
    
    print(f"\nExpected 360° frame: ~{expected_360}")
    print(f"Testing frames {start_idx} to {end_idx-1} (last {N_FRAMES_TO_TEST} frames)...")
    
    results = []
    print("\nComparing frames to first frame...")
    for idx in tqdm(range(start_idx, end_idx), desc="Testing frames"):
        frame_path = os.path.join(TEST_MASKS_DIR, image_files[idx])
        frame_mask = np.array(Image.open(frame_path))
        
        abs_diff, rel_diff, first_count, frame_count = calculate_pixel_difference(first_mask, frame_mask)
        
        results.append({
            'frame_idx': idx,
            'frame_number': idx + 1,
            'white_pixels': frame_count,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'rel_diff_pct': rel_diff * 100
        })
    
    # Sort by relative difference (normalized)
    results_sorted = sorted(results, key=lambda x: x['rel_diff'])
    
    print("\n" + "=" * 70)
    print("TOP 10 MATCHES (sorted by similarity to first frame):")
    print("=" * 70)
    print(f"{'Frame #':>8} | {'White Pixels':>13} | {'Abs Diff':>11} | {'Rel Diff %':>11}")
    print("-" * 70)
    print(f"{'FIRST':>8} | {first_white_count:>13,} | {'-':>11} | {'-':>11}")
    print("-" * 70)
    
    for result in results_sorted[:10]:
        print(f"{result['frame_number']:>8} | {result['white_pixels']:>13,} | "
              f"{result['abs_diff']:>11,} | {result['rel_diff_pct']:>11.2f}")
    
    best = results_sorted[0]
    
    print("\n" + "=" * 70)
    print(f"🎯 BEST MATCH: Frame #{best['frame_number']} (index {best['frame_idx']})")
    print(f"   White pixels: {best['white_pixels']:,} (first: {first_white_count:,})")
    print(f"   Difference: {best['abs_diff']:,} pixels ({best['rel_diff_pct']:.2f}%)")
    print(f"   → Use {best['frame_number']} frames for one complete 360° rotation")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: White pixel counts
    ax1 = axes[0, 0]
    frame_numbers = [r['frame_number'] for r in results]
    white_pixels = [r['white_pixels'] for r in results]
    ax1.plot(frame_numbers, white_pixels, 'b-', linewidth=2, label='Frame white pixels')
    ax1.axhline(first_white_count, color='green', linestyle='--', linewidth=2, label=f'First frame ({first_white_count:,})')
    ax1.axvline(best['frame_number'], color='red', linestyle='--', linewidth=2, label=f'Best match (#{best["frame_number"]})')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('White Pixel Count')
    ax1.set_title('White Pixel Counts in Last Frames', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute difference
    ax2 = axes[0, 1]
    abs_diffs = [r['abs_diff'] for r in results]
    ax2.plot(frame_numbers, abs_diffs, 'r-', linewidth=2)
    ax2.axvline(best['frame_number'], color='green', linestyle='--', linewidth=2, label=f'Best match (#{best["frame_number"]})')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Absolute Difference (pixels)')
    ax2.set_title('Difference from First Frame', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Show first frame
    ax3 = axes[1, 0]
    ax3.imshow(first_mask, cmap='gray')
    ax3.set_title(f'First Frame (Frame #1)\\nWhite pixels: {first_white_count:,}', fontweight='bold')
    ax3.axis('off')
    
    # Plot 4: Show best match frame
    ax4 = axes[1, 1]
    best_frame_path = os.path.join(TEST_MASKS_DIR, image_files[best['frame_idx']])
    best_frame_mask = np.array(Image.open(best_frame_path))
    ax4.imshow(best_frame_mask, cmap='gray')
    ax4.set_title(f'Best Match (Frame #{best["frame_number"]})\\nWhite pixels: {best["white_pixels"]:,}\\nDiff: {best["abs_diff"]:,} ({best["rel_diff_pct"]:.2f}%)', 
                  fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle(f'360° Frame Detection by Pixel Similarity (Total: {total_frames} frames)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 Action:")
    frames_to_remove = total_frames - best['frame_number']
    if frames_to_remove > 0:
        print(f"   Remove the last {frames_to_remove} frame(s) from your dataset")
        print(f"   This will give you exactly {best['frame_number']} frames = 360°")
    else:
        print(f"   Your dataset already represents 360° with {best['frame_number']} frames")


if __name__ == "__main__":
    main()
