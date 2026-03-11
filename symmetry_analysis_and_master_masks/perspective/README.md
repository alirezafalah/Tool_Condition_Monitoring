# Perspective Tilted Masks Module

Select tool mask folders from `DATA/masks/`, estimate perspective tilt, export corrected masks as PNG to `DATA/masks_tilted/`, and generate publication-quality debug figures and ROI visualizations.

## Files

| File | Purpose |
|---|---|
| `run_perspective_tools_gui.py` | GUI with three tabs: Tilt Processing, ROI Visualization, and Optimal Offset |
| `build_master_masks_all_two_edge_tools.py` | Core processor (CLI + importable) |
| `find_optimal_offset.py` | Optimal offset analysis helpers used by GUI Tab 3 |
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

### Tab 3 — Optimal Offset

1. Point **masks_tilted folder** to `DATA/masks_tilted` and click **Load Tools**.
2. Select one or more tool folders.
3. Choose **Mode**:
- `search_offset`: find best offset in a min/max range.
- `fixed_ranges`: compare two user-defined frame ranges directly (no searching).
4. Configure parameters:
- `Frames` + `Offset Min/Max` (search mode)
- `Range A start-end` + `Range B start-end` (fixed-range fallback)
- `Regions List` for multi-region fixed mode (example: `0-60,120-180,240-300`)
- `Use ROI Height from metadata JSON` (default): loads `roi_height_px` from `*_tilt_metadata.json` when available
- `ROI Height(px)` manual value: used if metadata is disabled or missing
- Output format(s): `PNG`, `SVG`, `PDF`
- Font sizes: title/axis/tick/legend
- `Include Top Caption` (off removes top caption)
- `Manual Legend Degree Ranges` (fixed-range mode): override displayed degree labels
  - `Legend Ranges List` supports many regions (example: `0-60,180-240,360-420`)
  - Default fixed-range display labels remain canonical per region: `0-N`, `180-(180+N)`, `360-(360+N)`, ...
5. Click **Run Optimal Offset Analysis**.
6. Outputs are written to each tool's `information/` directory.

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
        {tool_id}_optimal_offset_search.csv   # Offset sweep metrics table
        {tool_id}_optimal_offset_search_metadata.json # Offset analysis configuration + summary
        {tool_id}_optimal_offset_search.{png|svg|pdf} # Plots in selected formats
        {tool_id}_optimal_offset_search_abs_diff_per_angle.csv # Real frame index abs-diff rows for best offset
        {tool_id}_optimal_offset_search_pair_summary.csv # Pair-level summary for best offset
        {tool_id}_fixed_region_comparison.csv # Fixed-range comparison metrics table
        {tool_id}_fixed_region_comparison_metadata.json # Fixed-range config + summary
        {tool_id}_fixed_region_comparison.{png|svg|pdf} # Fixed-range plots in selected formats
        {tool_id}_fixed_region_comparison_abs_diff_per_angle.csv # Abs diff for each compared angle pair
        {tool_id}_fixed_region_comparison_region_counts.csv # Per-region counts by pair index
        {tool_id}_fixed_region_comparison_pair_summary.csv # Mean/max abs diff by region pair
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
