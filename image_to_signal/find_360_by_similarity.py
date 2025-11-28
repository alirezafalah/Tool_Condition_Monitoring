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
    """Find which frame matches the first TWO frames best (sequential consistency)."""
    print("=" * 70)
    print("FIND 360° BY SEQUENTIAL FRAME SIMILARITY")
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
    
    # Load first TWO frames
    first_path = os.path.join(TEST_MASKS_DIR, image_files[0])
    second_path = os.path.join(TEST_MASKS_DIR, image_files[1])
    first_mask = np.array(Image.open(first_path))
    second_mask = np.array(Image.open(second_path))
    first_white_count = count_white_pixels(first_mask)
    second_white_count = count_white_pixels(second_mask)
    
    print(f"First frame white pixels: {first_white_count:,}")
    print(f"Second frame white pixels: {second_white_count:,}")
    
    # Expected range for 360° (based on 12.2s @ 5 RPM = 366°)
    expected_360 = int((360.0 / 366.0) * total_frames)
    
    # Test last N frames
    N_FRAMES_TO_TEST = 25
    start_idx = max(expected_360 - 15, total_frames - N_FRAMES_TO_TEST)
    end_idx = min(total_frames, expected_360 + 15)
    
    print(f"\nExpected 360° frame: ~{expected_360}")
    print(f"Testing frames {start_idx} to {end_idx-2} (checking pairs)...")
    
    results = []
    print("\nComparing frame PAIRS to first two frames...")
    # Test pairs: (i, i+1) should match (0, 1)
    for idx in tqdm(range(start_idx, end_idx - 1), desc="Testing frame pairs"):
        frame_path = os.path.join(TEST_MASKS_DIR, image_files[idx])
        next_frame_path = os.path.join(TEST_MASKS_DIR, image_files[idx + 1])
        
        frame_mask = np.array(Image.open(frame_path))
        next_frame_mask = np.array(Image.open(next_frame_path))
        
        # Compare frame[i] with first frame
        abs_diff1, rel_diff1, _, frame_count = calculate_pixel_difference(first_mask, frame_mask)
        
        # Compare frame[i+1] with second frame
        abs_diff2, rel_diff2, _, next_frame_count = calculate_pixel_difference(second_mask, next_frame_mask)
        
        # Combined score (average of both relative differences)
        combined_score = (rel_diff1 + rel_diff2) / 2
        
        results.append({
            'frame_idx': idx,
            'frame_number': idx + 1,
            'white_pixels': frame_count,
            'next_white_pixels': next_frame_count,
            'abs_diff1': abs_diff1,
            'abs_diff2': abs_diff2,
            'rel_diff1': rel_diff1,
            'rel_diff2': rel_diff2,
            'combined_score': combined_score,
            'combined_score_pct': combined_score * 100
        })
    
    # Sort by combined score (lower is better)
    results_sorted = sorted(results, key=lambda x: x['combined_score'])
    
    print("\n" + "=" * 70)
    print("TOP 10 MATCHES (sorted by combined similarity):")
    print("=" * 70)
    print(f"{'Frame Pair':>11} | {'Frame i':>13} | {'Frame i+1':>13} | {'Score %':>10}")
    print("-" * 70)
    print(f"{'REFERENCE':>11} | {first_white_count:>13,} | {second_white_count:>13,} | {'-':>10}")
    print("-" * 70)
    
    for result in results_sorted[:10]:
        pair_str = f"{result['frame_number']}-{result['frame_number']+1}"
        print(f"{pair_str:>11} | {result['white_pixels']:>13,} | "
              f"{result['next_white_pixels']:>13,} | {result['combined_score_pct']:>10.2f}")
    
    best = results_sorted[0]
    
    print("\n" + "=" * 70)
    print(f"🎯 BEST MATCH: Frames #{best['frame_number']} and #{best['frame_number']+1}")
    print(f"   Frame {best['frame_number']}: {best['white_pixels']:,} pixels (ref: {first_white_count:,})")
    print(f"   Frame {best['frame_number']+1}: {best['next_white_pixels']:,} pixels (ref: {second_white_count:,})")
    print(f"   Diff 1st pair: {best['rel_diff1']*100:.2f}%, Diff 2nd pair: {best['rel_diff2']*100:.2f}%")
    print(f"   Combined score: {best['combined_score_pct']:.2f}%")
    print(f"   → Use {best['frame_number']} frames for one complete 360° rotation")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: White pixel counts for both pairs
    ax1 = axes[0, 0]
    frame_numbers = [r['frame_number'] for r in results]
    white_pixels_1 = [r['white_pixels'] for r in results]
    white_pixels_2 = [r['next_white_pixels'] for r in results]
    ax1.plot(frame_numbers, white_pixels_1, 'b-', linewidth=2, label='Frame i')
    ax1.plot([r['frame_number']+1 for r in results], white_pixels_2, 'c--', linewidth=2, label='Frame i+1')
    ax1.axhline(first_white_count, color='green', linestyle='--', linewidth=1.5, label=f'Ref frame 1 ({first_white_count:,})')
    ax1.axhline(second_white_count, color='lime', linestyle='--', linewidth=1.5, label=f'Ref frame 2 ({second_white_count:,})')
    ax1.axvline(best['frame_number'], color='red', linestyle='-', linewidth=2, label=f'Best match (#{best["frame_number"]})')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('White Pixel Count')
    ax1.set_title('White Pixel Counts - Sequential Pairs', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined scores
    ax2 = axes[0, 1]
    combined_scores = [r['combined_score_pct'] for r in results]
    ax2.plot(frame_numbers, combined_scores, 'r-', linewidth=2)
    ax2.axvline(best['frame_number'], color='green', linestyle='--', linewidth=2, label=f'Best match (#{best["frame_number"]})')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Combined Score (%)')
    ax2.set_title('Sequential Match Quality (lower = better)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Show reference frames (1 and 2)
    ax3 = axes[1, 0]
    # Show both reference frames side by side
    ref_combined = np.hstack([first_mask, second_mask])
    ax3.imshow(ref_combined, cmap='gray')
    ax3.set_title(f'Reference: Frame #1 (left) and #2 (right)\\n{first_white_count:,} | {second_white_count:,} pixels', 
                  fontweight='bold', fontsize=10)
    ax3.axis('off')
    
    # Plot 4: Show best match frames
    ax4 = axes[1, 1]
    best_frame_path = os.path.join(TEST_MASKS_DIR, image_files[best['frame_idx']])
    next_best_frame_path = os.path.join(TEST_MASKS_DIR, image_files[best['frame_idx'] + 1])
    best_frame_mask = np.array(Image.open(best_frame_path))
    next_best_frame_mask = np.array(Image.open(next_best_frame_path))
    best_combined = np.hstack([best_frame_mask, next_best_frame_mask])
    ax4.imshow(best_combined, cmap='gray')
    ax4.set_title(f'Best Match: Frame #{best["frame_number"]} (left) and #{best["frame_number"]+1} (right)\\n{best["white_pixels"]:,} | {best["next_white_pixels"]:,} pixels\\nScore: {best["combined_score_pct"]:.2f}%', 
                  fontweight='bold', fontsize=10)
    ax4.axis('off')
    
    plt.suptitle(f'360° Detection by Sequential Frame Similarity (Total: {total_frames} frames)', 
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
