"""
Rotate tool mask images by a fixed angle, then run left-right symmetry plots.

This script:
- Optionally rotates masks for selected tools by ROTATION_ANGLE_DEG
- Saves rotated copies to a new folder (if rotation enabled)
- Runs the same analysis/plots on the rotated masks
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")

TOOL_IDS = ["tool012", "tool115"]
ROTATION_ANGLE_DEG = -1.0  # Set to 0.0 to disable rotation

START_FRAME = 0
NUM_FRAMES = 90
ROI_HEIGHT = 200

OUTPUT_FORMATS = ["png"]

# Rotated masks destination base (only used if ROTATION_ANGLE_DEG != 0.0)
ROTATED_MASKS_DIR = os.path.join(BASE_DIR, "masks_rotated", f"rot_{ROTATION_ANGLE_DEG}deg")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", f"left_right_method_rot_{ROTATION_ANGLE_DEG}deg")


def get_mask_folder(tool_id):
	"""Get the mask folder path for a tool, trying different naming patterns."""
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
	"""Get all mask files from a folder, sorted by frame number/degree."""
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


def read_mask(path):
	"""Read mask image as grayscale; returns None if unreadable."""
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	if img is not None:
		return img
	try:
		import imageio.v2 as imageio

		img = imageio.imread(path)
		if img.ndim == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img
	except Exception:
		return None


def rotate_image(image, angle_deg):
	"""Rotate image around center, preserving size."""
	h, w = image.shape[:2]
	center = (w / 2.0, h / 2.0)
	mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	return cv2.warpAffine(
		image,
		mat,
		(w, h),
		flags=cv2.INTER_NEAREST,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=0,
	)


def rotate_and_save_masks(tool_id):
	"""Rotate all masks for a tool and save to ROTATED_MASKS_DIR."""
	src_folder = get_mask_folder(tool_id)
	if not src_folder:
		raise FileNotFoundError(f"No mask folder found for {tool_id}.")

	files = get_mask_files(src_folder)
	if not files:
		raise FileNotFoundError(f"No mask files found in {src_folder}.")

	dst_folder_name = os.path.basename(src_folder)
	dst_folder = os.path.join(ROTATED_MASKS_DIR, dst_folder_name)
	os.makedirs(dst_folder, exist_ok=True)

	for path in files:
		img = read_mask(path)
		if img is None:
			continue
		rotated = rotate_image(img, ROT_ANGLE_DEG)
		filename = os.path.basename(path)
		out_path = os.path.join(dst_folder, filename)
		cv2.imwrite(out_path, rotated)

	return dst_folder


def find_global_roi_bottom(mask_files, num_frames):
	"""Find the most bottom white pixel across all frames (global ROI)."""
	global_bottom = 0
	for mask_path in mask_files[:num_frames]:
		mask = read_mask(mask_path)
		if mask is None:
			continue
		white_pixels = np.where(mask == 255)
		if len(white_pixels[0]) > 0:
			bottom_row = np.max(white_pixels[0])
			global_bottom = max(global_bottom, bottom_row)
	return global_bottom


def analyze_left_right_symmetry(mask_path, global_roi_bottom, roi_height):
	"""Analyze a single mask image for left-right symmetry."""
	mask = read_mask(mask_path)
	if mask is None:
		return None

	roi_top = max(0, global_roi_bottom - roi_height)
	roi_bottom = global_roi_bottom + 1
	roi_mask = mask[roi_top:roi_bottom, :]

	white_pixels = np.where(roi_mask == 255)
	if len(white_pixels[1]) == 0:
		return None

	left_col = np.min(white_pixels[1])
	right_col = np.max(white_pixels[1])
	center_col = (left_col + right_col) // 2

	left_half = roi_mask[:, left_col:center_col]
	right_half = roi_mask[:, center_col + 1:right_col + 1]

	left_count = np.sum(left_half == 255)
	right_count = np.sum(right_half == 255)

	diff = abs(left_count - right_count)
	total = left_count + right_count
	ratio = diff / total if total > 0 else 0

	half_width = center_col - left_col
	half_area = roi_height * half_width if half_width > 0 else 1
	normalized_diff = diff / half_area

	return {
		"left_count": left_count,
		"right_count": right_count,
		"difference": diff,
		"ratio": ratio,
		"normalized_diff": normalized_diff,
		"center_col": center_col,
	}


def plot_analysis_results(results_df, tool_id, output_dir):
	"""Plot left-right symmetry analysis results."""
	plt.rcParams.update(
		{
			"font.size": 12,
			"axes.titlesize": 14,
			"axes.labelsize": 12,
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
			"legend.fontsize": 10,
		}
	)

	fig, axes = plt.subplots(2, 2, figsize=(14, 10))

	ax1 = axes[0, 0]
	ax1.plot(results_df["Frame"], results_df["Left Count"], label="Left Half", color="blue", alpha=0.7, linewidth=1.5)
	ax1.plot(results_df["Frame"], results_df["Right Count"], label="Right Half", color="red", alpha=0.7, linewidth=1.5)
	ax1.set_title(f"{tool_id}: White Pixel Count per Half")
	ax1.set_xlabel("Frame (Degrees)")
	ax1.set_ylabel("White Pixel Count")
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	ax2 = axes[0, 1]
	ax2.plot(results_df["Frame"], results_df["Difference"], color="purple", linewidth=1.5)
	ax2.fill_between(results_df["Frame"], results_df["Difference"], color="purple", alpha=0.2)
	ax2.set_title("Absolute Difference (Left - Right)")
	ax2.set_xlabel("Frame (Degrees)")
	ax2.set_ylabel("Pixel Count Difference")
	ax2.grid(True, alpha=0.3)

	ax3 = axes[1, 0]
	ax3.plot(results_df["Frame"], results_df["Ratio"], color="green", linewidth=1.5)
	ax3.set_title("Asymmetry Ratio (|L-R| / Total)")
	ax3.set_xlabel("Frame (Degrees)")
	ax3.set_ylabel("Ratio")
	ax3.grid(True, alpha=0.3)

	ax4 = axes[1, 1]
	ax4.plot(results_df["Frame"], results_df["Normalized Diff"], color="orange", linewidth=1.5)
	ax4.set_title("Normalized Difference (by half area)")
	ax4.set_xlabel("Frame (Degrees)")
	ax4.set_ylabel("Normalized Difference")
	ax4.grid(True, alpha=0.3)

	mean_ratio = results_df["Ratio"].mean()
	max_ratio = results_df["Ratio"].max()
	fig.suptitle(
		f"{tool_id} Left-Right Symmetry Analysis (rot {ROTATION_ANGLE_DEG} deg)\n"
		f"Mean Asymmetry Ratio: {mean_ratio:.4f}, Max: {max_ratio:.4f}",
		fontsize=14,
		fontweight="bold",
	)

	plt.tight_layout()

	for fmt in OUTPUT_FORMATS:
		path = os.path.join(output_dir, f"{tool_id}_left_right_analysis_rot_{ROTATION_ANGLE_DEG}deg.{fmt}")
		plt.savefig(path, format=fmt, dpi=300)

	plt.close()


def plot_sample_frames(mask_files, global_roi_bottom, roi_height, tool_id, output_dir):
	"""Plot sample frames showing the ROI guide lines."""
	sample_frames = [0, 22, 45, 67, 89]
	n_samples = len(sample_frames)
	fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 6))
	if n_samples == 1:
		axes = [axes]

	roi_top = max(0, global_roi_bottom - roi_height)

	for idx, frame_idx in enumerate(sample_frames):
		if frame_idx >= len(mask_files):
			continue
		mask = read_mask(mask_files[frame_idx])
		if mask is None:
			continue

		height, width = mask.shape
		roi_mask = mask[roi_top:global_roi_bottom + 1, :]
		white_pixels = np.where(roi_mask == 255)
		if len(white_pixels[1]) > 0:
			left_col = np.min(white_pixels[1])
			right_col = np.max(white_pixels[1])
			center_col = (left_col + right_col) // 2
		else:
			center_col = width // 2
			left_col = 0
			right_col = width - 1

		vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
		cv2.line(vis, (0, roi_top), (width, roi_top), (0, 255, 0), 2)
		cv2.line(vis, (0, global_roi_bottom), (width, global_roi_bottom), (0, 255, 0), 2)
		cv2.line(vis, (left_col, roi_top), (left_col, global_roi_bottom), (255, 255, 0), 1)
		cv2.line(vis, (right_col, roi_top), (right_col, global_roi_bottom), (255, 255, 0), 1)
		cv2.line(vis, (center_col, roi_top), (center_col, global_roi_bottom), (255, 0, 0), 2)

		axes[idx].imshow(vis)
		axes[idx].set_title(f"Frame {frame_idx}°\nCenter: col {center_col}")
		axes[idx].axis("off")

	fig.suptitle(
		f"{tool_id}: Sample Frames (rot {ROTATION_ANGLE_DEG} deg)\nROI (green), Tool bounds (yellow), Center (red)",
		fontsize=12,
	)
	plt.tight_layout()

	for fmt in OUTPUT_FORMATS:
		path = os.path.join(output_dir, f"{tool_id}_sample_frames_rot_{ROTATION_ANGLE_DEG}deg.{fmt}")
		plt.savefig(path, format=fmt, dpi=300)

	plt.close()


def analyze_tool(tool_id, mask_folder):
	"""Analyze a tool's rotated masks and save plots."""
	mask_files = get_mask_files(mask_folder)
	if not mask_files:
		return

	end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))
	actual_num_frames = end_frame - START_FRAME
	if actual_num_frames < 10:
		return

	global_roi_bottom = find_global_roi_bottom(mask_files, actual_num_frames)
	results = []

	for i in range(START_FRAME, end_frame):
		result = analyze_left_right_symmetry(mask_files[i], global_roi_bottom, ROI_HEIGHT)
		if result:
			results.append(
				{
					"Frame": i,
					"Left Count": result["left_count"],
					"Right Count": result["right_count"],
					"Difference": result["difference"],
					"Ratio": result["ratio"],
					"Normalized Diff": result["normalized_diff"],
					"Center Col": result["center_col"],
				}
			)

	if not results:
		return

	results_df = pd.DataFrame(results)
	plot_analysis_results(results_df, tool_id, OUTPUT_DIR)
	plot_sample_frames(mask_files, global_roi_bottom, ROI_HEIGHT, tool_id, OUTPUT_DIR)


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	if ROTATION_ANGLE_DEG != 0.0:
		os.makedirs(ROTATED_MASKS_DIR, exist_ok=True)

	for tool_id in TOOL_IDS:
		if ROTATION_ANGLE_DEG != 0.0:
			print(f"Rotating {tool_id} masks by {ROTATION_ANGLE_DEG} deg...")
			rotated_folder = rotate_and_save_masks(tool_id)
			print(f"Saved rotated masks to: {rotated_folder}")
		else:
			print(f"Using original {tool_id} masks (rotation disabled)...")
			src_folder = get_mask_folder(tool_id)
			if not src_folder:
				raise FileNotFoundError(f"No mask folder found for {tool_id}.")
			rotated_folder = src_folder

		print(f"Running analysis for {tool_id}...")
		analyze_tool(tool_id, rotated_folder)

	print("Done.")


if __name__ == "__main__":
	main()
