# Tool Condition Monitoring

This repository provides a complete pipeline for analyzing CNC tool wear through automated image processing and a professional GUI for visualization and metadata management.

## Project Structure

* **`image_to_signal/`**: Main processing pipeline that converts tool images into 1D profile signals for wear analysis
* **`mask_refiner/`**: Standalone interactive OpenGL mask refinement tool for editing masks against blurred frames
* **`tool_profile_viz/`**: Modern PyQt6 GUI application for visualizing tool profiles, managing metadata, and reviewing inspection results
* **`signal_processing/`** and **`tool_monitoring/`**: Legacy versions, kept for historical reference
* **`old/`**: Proof of concept implementations and experimental data

## Features

### Image Processing Pipeline
- Automated blur detection and preprocessing
- Background subtraction with multiple methods
- Contour detection and profile extraction
- 1D signal generation (area vs. rotation angle)
- Batch processing with progress tracking

### Visualization GUI
- Interactive matplotlib-based profile viewer
- Quarter-turn overview with synchronized image viewing
- ROI toggling for blurred/masked images
- Metadata management with undo/redo
- iOS-style toggle switches for clean UX
- Status column for inspection tracking (hidden by default to prevent accidental edits)

## Quick Start

### Run Image Processing Pipeline (GUI)
```bash
cd Tool_Condition_Monitoring
python -m image_to_signal.gui_main
```

### Run Mask Refiner (Standalone)
```bash
cd Tool_Condition_Monitoring
python -m mask_refiner.main
```

Optional startup with a tool preselected:
```bash
python -m mask_refiner.main --tool tool002
```

Optional custom DATA root:
```bash
python -m mask_refiner.main --data-dir "C:/path/to/DATA"
```

Inside the refiner you can either:
- pick a tool from the dropdown (auto-resolves `DATA/blurred` + `DATA/masks`), or
- manually choose `Blurred Folder` and `Masks Folder` from the Dataset panel.

### Run Profile Visualization Tool
```bash
cd Tool_Condition_Monitoring/tool_profile_viz
python main.py
```

### Find Optimal Frame Count for 360°
```bash
cd Tool_Condition_Monitoring
python -m image_to_signal.find360
```

### Rename Raw Images Using Detected Frame Count
After you know the 360° frame count (e.g. 363):
```bash
cd Tool_Condition_Monitoring
python -m image_to_signal.rename_by_angle --tool tool002 --frames360 363
```
Then run the GUI or pipeline steps.

### Launch Mask Refiner from Pipeline GUI
In `python -m image_to_signal.gui_main`, open the `🎭 Mask Refiner` tab and launch it either:
- with current Tool ID, or
- in manual selection mode.
