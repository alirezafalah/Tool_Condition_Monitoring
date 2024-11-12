# Tool Monitoring Project

This project contains methods for analyzing tool images and video frames to monitor tool integrity and create visualizations. The scripts provided are intended for tasks such as graph generation, image filtering, and collage creation for video frames.

## Directory Structure

- **scripts/** - Python scripts for various processing and analysis tasks.
- **data/** - Input data for analysis, including CSV files with tool metrics.
- **output/** - Directory for saving output frames, processed images, and collages.

## Setup

1. Clone or download this repository.
2. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib opencv-python
   ```
3. Ensure you have a `data` folder containing required CSV files, and an `output` folder for generated files.

## Usage

### 1. 360 Degree Graph Maker
**File:** `scripts/graph_maker_360degree.py`

Generates a side-by-side plot of intact and broken tool patterns. The script reads CSV files from the `data` directory and produces a plot for comparison.

- **Inputs:** `new_tool_pattern_from_broken.csv`, `table_broken.csv`
- **Output:** Visualization of tool patterns

### 2. Apply Filter
**File:** `scripts/filter_applier.py`

Applies various filters to enhance details, such as Canny edge detection and gamma correction. This can be used to analyze the region of interest (ROI) more effectively.

- **Inputs:** ROI image data
- **Output:** Filtered image for further processing

### 3. Background Threshold Closing
**File:** `scripts/background_threshold_closing.py`

Calculates the median background image and performs background subtraction. Ideal for isolating the tool from the background.

- **Inputs:** Background frames and tool frames
- **Output:** Processed images with background subtracted

### 4. Collage Maker
**File:** `scripts/collage_maker.py`

Extracts frames from a video, applies a mask, and combines selected frames into a collage.

- **Inputs:** Video file (e.g., `IMG_3619.MOV`)
- **Output:** Collage of frames

### 5. Frame and ROI Capturing
**File:** `scripts/frame_and_roi_capture.py`

Extracts frames from a video within a specified time range, capturing only the ROI. This is useful for analyzing specific sections of the tool in motion.

- **Inputs:** Video file, start and end times, ROI coordinates
- **Output:** Frames saved in the `output/frames/` directory

### 6. Mask Refinement
**File:** `scripts/mask_refiner.py`

Refines the mask to isolate the tool from its background more accurately by filling enclosed regions.

- **Inputs:** Frames of the tool
- **Output:** Refined mask images

### 7. Hole Filling
**File:** `scripts/hole_filler.py`

Fills enclosed regions in the binary mask of the tool image, helping to improve visibility of the tool's structure.

- **Inputs:** Frames and median background image
- **Output:** Processed frames with filled regions

## Notes
- Update the paths in each script according to your local setup.
- To adjust parameters like thresholds or kernel sizes, edit the respective arguments within each script.

## Contact

For further information or questions, please contact me at al.r.falah@gmail.com.
