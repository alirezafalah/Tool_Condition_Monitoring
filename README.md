# Tool Condition Monitoring

This repository provides a complete pipeline for analyzing CNC tool wear through automated image processing and a professional GUI for visualization and metadata management.

## Project Structure

* **`image_to_signal/`**: Main processing pipeline that converts tool images into 1D profile signals for wear analysis
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

### Run Profile Visualization Tool
```bash
cd Tool_Condition_Monitoring/tool_profile_viz
python src/main.py
```

### Find Optimal Frame Count for 360°
```bash
cd Tool_Condition_Monitoring
python -m image_to_signal.find360
```
Edit `TOOL_ID` at top of `find360.py`.

### Rename Raw Images Using Detected Frame Count
After you know the 360° frame count (e.g. 363):
```bash
cd Tool_Condition_Monitoring
python -m image_to_signal.rename_by_angle --tool tool002 --frames360 363
```
Then run the GUI or pipeline steps.
