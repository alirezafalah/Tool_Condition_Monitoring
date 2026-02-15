"""
Interactive rotation finder for tool masks and raw images.

Purpose:
- Rotate a selected tool's frames with a slider/entry
- Toggle between mask and raw image view
- Keep ROI guide lines fixed (green) while the image rotates
- Let you verify a single rotation angle across all frames
"""

import os
import glob
from collections import OrderedDict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")

TOOL_ID = "tool115"
START_FRAME = 0
ROI_HEIGHT = 200

ANGLE_MIN = -10.0
ANGLE_MAX = 10.0
ANGLE_STEP = 0.1

# Prefer GPU path (UMat) when available
USE_UMAT = True

# Cache to reduce disk reads
CACHE_MAX = 64

# Candidate raw image folders (first found is used)
RAW_DIR_CANDIDATES = [
	("blurred", os.path.join(BASE_DIR, "blurred", f"{TOOL_ID}_blurred")),
	("tools", os.path.join(BASE_DIR, "tools", TOOL_ID)),
]


def get_mask_files(tool_id):
	"""Get all final mask files for a tool, sorted by frame number."""
	mask_folder = os.path.join(MASKS_DIR, f"{tool_id}gain10paperBG_final_masks")

	if not os.path.exists(mask_folder):
		mask_folder = os.path.join(MASKS_DIR, f"{tool_id}gain10_final_masks")
	if not os.path.exists(mask_folder):
		mask_folder = os.path.join(MASKS_DIR, f"{tool_id}_final_masks")

	if not os.path.exists(mask_folder):
		raise FileNotFoundError(
			f"Could not find mask folder for {tool_id}. Tried patterns in {MASKS_DIR}"
		)

	files = get_files_sorted(mask_folder)
	if not files:
		raise FileNotFoundError(f"No mask files found in {mask_folder}")

	return files, mask_folder


def get_files_sorted(folder_path):
	"""Get all tiff files from a folder, sorted by frame number/degree."""
	pattern = os.path.join(folder_path, "*.tiff")
	files = glob.glob(pattern)
	if not files:
		pattern = os.path.join(folder_path, "*.tif")
		files = glob.glob(pattern)

	if not files:
		return []

	def extract_frame_num(filepath):
		basename = os.path.basename(filepath)
		parts = basename.replace(".tiff", "").replace(".tif", "").split("_")
		for part in reversed(parts):
			if part.isdigit():
				return float(part)
			try:
				return float(part)
			except ValueError:
				continue
		return 0.0

	return sorted(files, key=extract_frame_num)


def get_raw_files():
	"""Pick the first raw folder that exists and has images."""
	for label, folder in RAW_DIR_CANDIDATES:
		if not os.path.exists(folder):
			continue
		files = get_files_sorted(folder)
		if files:
			return files, folder, label
	return [], None, None


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


def read_raw(path):
	"""Read raw image as RGB; returns None if unreadable."""
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is not None:
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	try:
		import imageio.v2 as imageio

		img = imageio.imread(path)
		if img.ndim == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		return img
	except Exception:
		return None


def cache_get(cache, key, loader):
	"""Simple LRU cache for images."""
	if key in cache:
		cache.move_to_end(key)
		return cache[key]
	img = loader(key)
	if img is None:
		return None
	cache[key] = img
	if len(cache) > CACHE_MAX:
		cache.popitem(last=False)
	return img


def find_global_roi_bottom(mask_files, start_frame, end_frame):
	"""Find the most bottom white pixel across selected frames (global ROI)."""
	global_bottom = 0

	for i in range(start_frame, end_frame):
		mask = read_mask(mask_files[i])
		if mask is None:
			continue
		white_pixels = np.where(mask == 255)
		if len(white_pixels[0]) > 0:
			bottom_row = int(np.max(white_pixels[0]))
			global_bottom = max(global_bottom, bottom_row)

	return global_bottom


def rotate_image(image, angle_deg, interpolation):
	"""Rotate image around center, preserving size."""
	h, w = image.shape[:2]
	center = (w / 2.0, h / 2.0)
	mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	if USE_UMAT:
		try:
			umat = cv2.UMat(image)
			rotated = cv2.warpAffine(
				umat,
				mat,
				(w, h),
				flags=interpolation,
				borderMode=cv2.BORDER_CONSTANT,
				borderValue=0,
			)
			return rotated.get()
		except Exception:
			pass
	return cv2.warpAffine(
		image,
		mat,
		(w, h),
		flags=interpolation,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=0,
	)


def find_tool_center_col(mask, roi_top, roi_bottom):
	"""Find tool center based on white pixels in ROI; returns None if empty."""
	roi = mask[roi_top:roi_bottom, :]
	white_pixels = np.where(roi == 255)
	if len(white_pixels[1]) == 0:
		return None
	left_col = int(np.min(white_pixels[1]))
	right_col = int(np.max(white_pixels[1]))
	return (left_col + right_col) // 2


def main():
	mask_files, mask_folder = get_mask_files(TOOL_ID)
	raw_files, raw_folder, raw_label = get_raw_files()

	if not raw_files:
		raw_folder = None
		raw_label = None

	end_frame = len(mask_files)
	if end_frame - START_FRAME < 1:
		raise RuntimeError("No frames available for the selected range.")

	global_roi_bottom = find_global_roi_bottom(mask_files, START_FRAME, end_frame)
	roi_top = max(0, global_roi_bottom - ROI_HEIGHT)
	roi_bottom = global_roi_bottom + 1

	# Initial state
	current_frame = START_FRAME
	current_angle = 0.0
	mode = "raw" if raw_files else "mask"

	mask_cache = OrderedDict()
	raw_cache = OrderedDict()

	def get_mask_for_frame(frame_idx):
		path = mask_files[frame_idx]
		return cache_get(mask_cache, path, read_mask)

	def get_raw_for_frame(frame_idx):
		if not raw_files:
			return None
		if frame_idx >= len(raw_files):
			return None
		path = raw_files[frame_idx]
		return cache_get(raw_cache, path, read_raw)

	# Load initial image
	base_mask = get_mask_for_frame(current_frame)
	if base_mask is None:
		raise RuntimeError(
			"Failed to read the first mask. If TIFF LZW is used, install imagecodecs."
		)

	base_raw = get_raw_for_frame(current_frame)
	if mode == "raw" and base_raw is None:
		mode = "mask"

	rotated_mask = rotate_image(base_mask, current_angle, cv2.INTER_NEAREST)
	center_col = find_tool_center_col(rotated_mask, roi_top, roi_bottom)

	if mode == "raw":
		rotated_view = rotate_image(base_raw, current_angle, cv2.INTER_LINEAR)
		cmap = None
	else:
		rotated_view = rotated_mask
		cmap = "gray"

	# Figure
	fig, ax = plt.subplots(figsize=(8, 8))
	plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.92)

	im = ax.imshow(rotated_view, cmap=cmap)

	mode_label = "RAW" if mode == "raw" else "MASK"
	src_label = raw_label if mode == "raw" else "masks"
	ax.set_title(
		f"{TOOL_ID} | Frame {current_frame} | Angle {current_angle:.2f} deg | {mode_label} ({src_label})"
	)
	ax.axis("off")

	# ROI guide lines
	_ = ax.axhline(roi_top, color="lime", linewidth=1.5)
	_ = ax.axhline(global_roi_bottom, color="lime", linewidth=1.5)

	# Center guide
	center_line = None
	if center_col is not None:
		center_line = ax.axvline(center_col, color="red", linewidth=1.5)

	# Widgets
	ax_frame = plt.axes([0.08, 0.14, 0.78, 0.03])
	frame_slider = Slider(
		ax=ax_frame,
		label="Frame",
		valmin=START_FRAME,
		valmax=end_frame - 1,
		valinit=current_frame,
		valstep=1,
	)

	ax_angle = plt.axes([0.08, 0.09, 0.78, 0.03])
	angle_slider = Slider(
		ax=ax_angle,
		label="Angle (deg)",
		valmin=ANGLE_MIN,
		valmax=ANGLE_MAX,
		valinit=current_angle,
		valstep=ANGLE_STEP,
	)

	ax_text = plt.axes([0.08, 0.03, 0.30, 0.04])
	angle_text = TextBox(ax_text, "Set Angle", initial=str(current_angle))

	ax_reset = plt.axes([0.42, 0.03, 0.12, 0.04])
	reset_button = Button(ax_reset, "Reset")

	ax_toggle = plt.axes([0.56, 0.03, 0.12, 0.04])
	toggle_button = Button(ax_toggle, "Toggle")

	ax_stepm = plt.axes([0.71, 0.03, 0.08, 0.04])
	step_minus = Button(ax_stepm, "-0.5")

	ax_stepp = plt.axes([0.80, 0.03, 0.08, 0.04])
	step_plus = Button(ax_stepp, "+0.5")

	def update_title(frame_idx, angle_deg, view_mode):
		mode_label = "RAW" if view_mode == "raw" else "MASK"
		src_label = raw_label if view_mode == "raw" else "masks"
		ax.set_title(
			f"{TOOL_ID} | Frame {frame_idx} | Angle {angle_deg:.2f} deg | {mode_label} ({src_label})"
		)

	def update_view(frame_idx, angle_deg, view_mode):
		nonlocal center_line
		frame_idx = int(frame_idx)
		mask = get_mask_for_frame(frame_idx)
		if mask is None:
			return

		rot_mask = rotate_image(mask, angle_deg, cv2.INTER_NEAREST)
		new_center = find_tool_center_col(rot_mask, roi_top, roi_bottom)
		if center_line is not None:
			center_line.remove()
			center_line = None
		if new_center is not None:
			center_line = ax.axvline(new_center, color="red", linewidth=1.5)

		if view_mode == "raw":
			raw = get_raw_for_frame(frame_idx)
			if raw is None:
				return
			rot_view = rotate_image(raw, angle_deg, cv2.INTER_LINEAR)
			im.set_cmap(None)
		else:
			rot_view = rot_mask
			im.set_cmap("gray")

		im.set_data(rot_view)
		update_title(frame_idx, angle_deg, view_mode)
		fig.canvas.draw_idle()

	def on_frame_change(val):
		update_view(val, angle_slider.val, mode)

	def on_angle_change(val):
		update_view(frame_slider.val, val, mode)

	def on_text_submit(text):
		try:
			val = float(text)
		except ValueError:
			return
		val = max(ANGLE_MIN, min(ANGLE_MAX, val))
		angle_slider.set_val(val)

	def on_reset(_event):
		angle_slider.set_val(0.0)

	def on_step(delta):
		angle_slider.set_val(angle_slider.val + delta)

	def on_toggle(_event):
		nonlocal mode
		if mode == "mask" and raw_files:
			mode = "raw"
		else:
			mode = "mask"
		update_view(frame_slider.val, angle_slider.val, mode)

	frame_slider.on_changed(on_frame_change)
	angle_slider.on_changed(on_angle_change)
	angle_text.on_submit(on_text_submit)
	reset_button.on_clicked(on_reset)
	toggle_button.on_clicked(on_toggle)
	step_minus.on_clicked(lambda _evt: on_step(-0.5))
	step_plus.on_clicked(lambda _evt: on_step(0.5))

	def on_key(event):
		if event.key == "left":
			on_step(-0.1)
		elif event.key == "right":
			on_step(0.1)
		elif event.key == "up":
			frame_slider.set_val(min(frame_slider.val + 1, end_frame - 1))
		elif event.key == "down":
			frame_slider.set_val(max(frame_slider.val - 1, START_FRAME))
		elif event.key == "m":
			mode = "mask"
			update_view(frame_slider.val, angle_slider.val, mode)
		elif event.key == "r":
			if raw_files:
				mode = "raw"
				update_view(frame_slider.val, angle_slider.val, mode)

	fig.canvas.mpl_connect("key_press_event", on_key)

	plt.show()


if __name__ == "__main__":
	main()
