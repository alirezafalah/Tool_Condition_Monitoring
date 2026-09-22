# Image to Signal Utilities (`image_to_signal/utils`)

This directory contains utility modules, batch processing scripts, and interactive tools supporting the tool wear mask generation and analysis pipeline.

---

## 📋 Quick Directory Overview

| Script | Type | Description |
|---|---|---|
| [`mask_refiner.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/mask_refiner.py) | **GUI Tool** | Interactive PyQt6 mask editor with drawing brush, overlay comparison, zoom, and undo/redo. |
| [`mask_cleaner.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/mask_cleaner.py) | **CLI Batch Tool** | Automated mask post-processor to fill enclosed holes and remove error dots (keep largest contour). |
| [`mask_refinement.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/mask_refinement.py) | **Sandbox** | Multi-channel color threshold exploration script (LAB + HSV color spaces). |
| [`filters.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/filters.py) | **Core Module** | Core algorithms for background subtraction, multi-channel color masking, and morphological filtering. |
| [`optimized_processing.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/optimized_processing.py) | **Core Module** | Multi-mode execution engine (GPU via OpenCL, multi-core CPU, single-core fallback). |
| [`image_utils.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/image_utils.py) | **Helper** | Plotting, visualization, and side-by-side comparison helpers. |
| [`background_subtraction_playground.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/background_subtraction_playground.py) | **Sandbox** | Playground for evaluating background subtraction techniques. |
| [`degree_name_fixer.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/degree_name_fixer.py) | **Helper** | Utility to rename/standardize angular image file naming formats. |

---

## 🖌️ Interactive Mask Refiner (`mask_refiner.py`)

[`mask_refiner.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/mask_refiner.py) is a standalone desktop GUI for inspecting and manually touching up binary masks against reference masks or original frames.

### Features
- **Brush Modes**:
  - **Left Click**: Draw white (fill mask / add foreground).
  - **Right Click**: Draw black (erase mask / remove artifacts).
- **History (Undo / Redo)**:
  - Full stroke-by-stroke undo and redo history (up to 50 steps).
  - Unsaved modifications indicator (`*` on the Save button). Undoing back to the clean saved state automatically clears the unsaved status.
  - Confirmation prompt when switching masks or closing the window with unsaved changes.
- **Reference Overlay**:
  - Set any mask as a red reference overlay.
  - Real-time opacity slider (0–100%).
- **Navigation & Inspection**:
  - Mouse scroll for zooming in/out.
  - Reset Zoom button.
  - Real-time brush cursor indicator showing current radius.
  - Natural numerical ordering of frames in the list.

### Keyboard Shortcuts
| Shortcut | Action |
|---|---|
| `Ctrl + Z` | Undo last stroke |
| `Ctrl + Y` or `Ctrl + Shift + Z` | Redo undone stroke |
| `Ctrl + S` | Save mask to disk |
| `Mouse Scroll` | Zoom in / Zoom out |

### How to Run
```bash
# From the repository root:
python -m image_to_signal.utils.mask_refiner

# Or from within image_to_signal:
python utils/mask_refiner.py

# Or directly from the utils directory:
cd image_to_signal/utils && python mask_refiner.py
```

---

## 🧹 Batch Mask Cleaner (`mask_cleaner.py`)

[`mask_cleaner.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/mask_cleaner.py) is a high-throughput batch post-processing utility designed to automatically clean binary masks across an entire folder.

### Core Operations
1. **Fill Holes (`--fill-holes`)**:
   - Detects all enclosed background regions inside the tool silhouette and fills them solid (bucket-fill effect).
   - Fixes dark specular reflections, illumination shadows, or thresholding dropouts inside the tool body.
2. **Keep Largest Contour (`--keep-largest`)**:
   - Identifies all connected components and keeps only the primary (largest area) contour.
   - Completely removes flying chips, dust particles, stray threshold specks, and background noise dots.
3. **Execution Order**:
   - When both are enabled (default), hole filling is applied first, followed by largest contour retention (**1. Fill holes $\rightarrow$ 2. Keep largest contour**).
   - This ensures internal holes in the main object are sealed before filtering out external noise dots.

### Command-Line Usage
```bash
python utils/mask_cleaner.py <input_dir> [options]
```

### Options and Toggles
| Option | Default | Description |
|---|---|---|
| `input_dir` | *(Required)* | Path to the directory containing binary masks (`.png`, `.tif`, `.tiff`, `.jpg`, etc.). |
| `-o`, `--output-dir` | `<input_dir>_cleaned` | Directory to save processed masks. Created automatically if it does not exist. |
| `--in-place` | `False` | Overwrites files directly in `input_dir` instead of creating a new output folder. |
| `--fill-holes` | `None` (enabled by default) | Toggles hole filling on. |
| `--keep-largest` | `None` (enabled by default) | Toggles largest contour retention on. |
| `--no-fill-holes` | `False` | Explicitly disables hole filling. |
| `--no-keep-largest` | `False` | Explicitly disables keeping largest contour. |
| `--order` | `fill-first` | Execution order: `fill-first` or `largest-first`. |
| `--threshold` | `127` | Binarization cutoff threshold (0–255). |
| `--dry-run` | `False` | Analyzes masks and displays summary statistics without writing files to disk. |
| `-j`, `--jobs` | CPU count | Number of parallel worker threads. |

### CLI Examples
```bash
# 1. Run both operations (fill holes, then remove error dots) into a new folder:
python utils/mask_cleaner.py /path/to/masks

# 2. Run only hole filling:
python utils/mask_cleaner.py /path/to/masks --fill-holes

# 3. Run only keeping largest contour (remove error dots):
python utils/mask_cleaner.py /path/to/masks --keep-largest

# 4. Save to a specific output folder:
python utils/mask_cleaner.py /path/to/masks -o /path/to/cleaned_masks

# 5. Overwrite masks directly in-place:
python utils/mask_cleaner.py /path/to/masks --in-place

# 6. Dry run to inspect statistics before applying changes:
python utils/mask_cleaner.py /path/to/masks --dry-run
```

### Python API Usage
You can also import functions from [`mask_cleaner.py`](file:///home/alifalah/Projects/Tool_Condition_Monitoring/image_to_signal/utils/mask_cleaner.py) into your own scripts or pipelines:

```python
import cv2
from image_to_signal.utils.mask_cleaner import clean_mask, fill_holes, keep_largest_contour

# Process a single mask in memory
mask = cv2.imread("mask_001.png", cv2.IMREAD_GRAYSCALE)
cleaned_mask, stats = clean_mask(
    mask,
    do_fill_holes=True,
    do_keep_largest=True,
    order="fill-first"
)

# Or batch process a whole folder
from image_to_signal.utils.mask_cleaner import process_mask_directory

summary = process_mask_directory(
    input_dir="/path/to/raw_masks",
    output_dir="/path/to/cleaned_masks",
    do_fill_holes=True,
    do_keep_largest=True
)
print(f"Processed {summary['success_count']} masks successfully.")
```
