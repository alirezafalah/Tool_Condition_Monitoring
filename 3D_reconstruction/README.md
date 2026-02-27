# 3D Reconstruction — Visual Hull (Shape from Silhouette)

GPU-accelerated 3D reconstruction of CNC cutting tools from binary mask
silhouettes, with a professional tkinter GUI.

## Quick Start

```bash
python visual_hull_gui.py
```

## Voxel Grid Specification (for ML)

All `.npz` files produced by this pipeline use a **fixed cubic grid** with a
**global physical bounding volume** so they can be batched directly in
TensorFlow / PyTorch with consistent spatial scale.

### Grid Layout

```
┌─────────────────────────────────────────────────┐
│                128 × 128 × 128                  │
│              (or user-chosen N³)                 │
│                                                  │
│   Physical volume: 20 × 20 × 20 mm (±10 mm)    │
│   Voxel size:      0.15625 mm  (20/128)         │
│                                                  │
│   Axes:   X = lateral  (left ↔ right)           │
│           Y = axial    (along tool shaft)        │
│           Z = depth    (front ↔ back)            │
│                                                  │
│   Origin: rotation axis centre in camera frame   │
│   Units:  millimetres (mm)                       │
└─────────────────────────────────────────────────┘
```

### Global fixed bounding cube (CRITICAL for ML)

**Every tool in the dataset is carved inside the exact same 20 mm cube.**
This guarantees:
- `volume_bounds` is **mathematically identical** in every `.npz` file
- The mm-per-voxel ratio is **constant** across all 1,800 tools
- The 3D CNN learns a consistent geometric scale

The cube is derived from the camera optics:

| Parameter | Value |
|-----------|-------|
| Lens | VS-LDV75, f = 75 mm |
| Sensor | 1/2" (6.4 × 4.8 mm) |
| Working distance | 250 mm |
| Visible width | 21.33 mm |
| Visible height | 16.00 mm |
| **Global cube** | **±10 mm (20 mm)** |
| Projection scale | `75 × img_w / (6.4 × 250)` px/mm |

### How voxel carving works

1. **Silhouette analysis** — sample masks are loaded to find the rotation
   axis centre (`cx`, `cy`) in pixel coordinates.

2. **Camera-based projection** — the projection scale (px/mm) is computed
   from the camera parameters, NOT from the tool's silhouette width.
   This is a **global constant** for the dataset.

3. **Fixed grid** — the 20 mm cube is discretised into 128³ voxels.
   All voxels start as `True` (occupied).

4. **Carving** — for each view, each voxel is projected onto the mask.
   If it falls **outside the silhouette** OR **outside the camera frame**,
   it's carved to `False`.

5. **Result** — smaller tools occupy fewer voxels; the surrounding empty
   space is naturally `False` (zero-padded). The grid shape and bounds
   are identical for every tool.

### NPZ file contents

| Array            | Shape        | Dtype     | Description |
|------------------|-------------|-----------|-------------|
| `voxel_grid`     | (N, N, N)   | `bool`    | Occupancy: `True` = inside tool |
| `volume_bounds`  | (3, 2)      | `float32` | `[[-10, 10], [-10, 10], [-10, 10]]` mm — **SAME for all tools** |
| `grid_shape`     | (3,)        | `int32`   | `[128, 128, 128]` — always matches `voxel_grid.shape` |

### Loading example

```python
import numpy as np

data = np.load("tool_visual_hull_voxels.npz")
grid   = data["voxel_grid"]       # (128, 128, 128), dtype bool
bounds = data["volume_bounds"]    # [[-10, 10], [-10, 10], [-10, 10]] mm — SAME for all

# Physical coordinate of voxel [i, j, k]:
x_mm = -10.0 + i * 20.0 / 128   # = -10 + i * 0.15625
y_mm = -10.0 + j * 20.0 / 128
z_mm = -10.0 + k * 20.0 / 128

# For TF / PyTorch — add channel dim
tensor = grid[np.newaxis, ...].astype(np.float32)  # (1, 128, 128, 128)
```

### Memory budget (1,800 tools)

| Dtype     | Per tool | 1,800 tools |
|-----------|----------|-------------|
| `bool`    | 2.1 MB   | **3.8 GB**  |
| `float32` | 8.4 MB   | 15.1 GB     |

The pipeline saves `bool` by default so that the full dataset fits in 16 GB
RAM during training.

### Canonical spec file

See [`voxel_grid_spec.json`](voxel_grid_spec.json) for the machine-readable
specification including camera parameters, global bounds, and loading examples.
Use this in downstream scripts (ground truth generation, data loaders,
evaluation) to ensure grid shape and bounds are interpreted consistently.

## Per-tool output

Each reconstruction also saves a `run_config.json` in the tool's output folder
containing:

- `grid_shape` — `[128, 128, 128]` (same for all tools)
- `volume_bounds` — `[[-10, 10], [-10, 10], [-10, 10]]` mm (same for all tools)
- `voxel_dtype` — `"bool"`
- `occupied_voxels` / `total_voxels` / `occupancy_ratio`
- All engine parameters used for that run

## Files

| File | Description |
|------|-------------|
| `visual_hull_gui.py`   | tkinter GUI — launch this |
| `visual_hull_engine.py`| Core reconstruction engine (GPU + CPU) |
| `visual_hull.py`       | Legacy CLI entry point |
| `voxel_grid_spec.json` | Machine-readable voxel grid spec |
