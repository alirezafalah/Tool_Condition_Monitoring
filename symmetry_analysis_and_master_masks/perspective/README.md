# Perspective Tilted Masks Module

Select tool mask folders from `DATA/masks/`, estimate perspective tilt, export corrected masks as PNG to `DATA/masks_tilted/`, and generate publication-quality debug figures and ROI visualizations.

## Files

| File | Purpose |
|---|---|
| `run_perspective_tools_gui.py` | Matrix-themed GUI with two tabs: Tilt Processing and ROI Visualization |
| `build_master_masks_all_two_edge_tools.py` | Core processor (CLI + importable) |
| `fix_perspective.py` | Legacy single-tool script (retained for compatibility) |

## Workflow

### Tab 1 — Tilt Processing

1. Set **Base DATA Directory** and **Masks Root Directory** (defaults to `DATA/masks`).
2. Click **Refresh Folders** to list available tool mask folders.
3. Select one, multiple, or all folders.
4. Set **Max Workers** (bounded parallel processing, 16-core friendly).
5. Click **Run Tilt Processing**.

### Tab 2 — ROI Visualization

1. Point **masks_tilted folder** to `DATA/masks_tilted`.
2. Click **Load Tools** to list processed tools.
3. Select a tool folder.
4. Optionally set **Frame Index** (blank = random) and **Caption Override** (blank = auto, e.g. "Drill Tool (062)", `-` = no caption).
5. Click **Generate ROI Figure**.

## Output Structure

For each tool, outputs are saved to:

```
DATA/masks_tilted/{tool_id}_final_masks/
    *.png                                     # Tilt-corrected binary masks
    information/
        {tool_id}_MASTER_MASK.png             # OR-combined master mask
        {tool_id}_angle_calculation.png       # Figure 1: tilt angle from master mask
        {tool_id}_centerline_determination.png # Figure 2: centerline on straightened mask
        {tool_id}_roi_visualization.png       # Figure 3: dynamic ROI with left/right coloring
        {tool_id}_tilt_metadata.json          # Per-tool calibration metadata
```

## Key Behaviors

- **All-white frame skipping**: Frames where the entire mask is white (255) are excluded from the master mask.
- **Bounded parallelism**: Default worker count reserves CPU headroom on multi-core systems.
- **OpenCL**: Enabled by default when available (Intel Iris Xe compatible).
- **Debug figures**: Two matplotlib figures per tool (tilt angle calculation + centerline determination), matching publication style.
- **ROI figure**: Dynamic ROI box (height = 0.45 × master mask width) centered on the geometric centerline, with left (blue) and right (red) halves, legend, and tool caption.

## Run

```powershell
py -3 "Tool_Condition_Monitoring\symmetry_analysis_and_master_masks\perspective\run_perspective_tools_gui.py"
```

## CLI (without GUI)

```powershell
py -3 "Tool_Condition_Monitoring\symmetry_analysis_and_master_masks\perspective\build_master_masks_all_two_edge_tools.py" `
  --base-data-dir "C:\path\to\CCD_DATA\DATA" `
  --max-workers 6 `
  --mask-folder "C:\path\to\DATA\masks\tool012_final_masks" `
  --mask-folder "C:\path\to\DATA\masks\tool028_final_masks"
```

Optional: `--disable-opencl` to force CPU-only OpenCV.
