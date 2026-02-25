#!/usr/bin/env python3
"""
Visual Hull (Shape from Silhouette) Reconstruction for Turntable Drill Tools
=============================================================================

Reconstructs a 3D voxel model from turntable silhouette masks of a drill tool.

Algorithm overview:
  1. Load binary silhouette masks with known rotation angles
  2. Estimate projection parameters from silhouette extents (rotation axis, scale)
  3. Create a 3D voxel grid enclosing the tool
  4. For each view, project voxels onto the image plane (orthographic projection)
  5. Carve voxels that project outside the silhouette in ANY view
  6. Export the remaining occupied voxels as mesh, point cloud, and/or voxel grid

Turntable geometry (orthographic model):
  - Y axis  = rotation axis (drill axis, vertical in image, pointing UP)
  - X, Z    = horizontal plane perpendicular to drill axis
  - Origin   = center of rotation on the drill axis

  At rotation angle θ, the orthographic projection of world point (X, Y, Z) is:
      x_proj =  X·cos(θ) + Z·sin(θ)     [horizontal in image]
      y_proj =  Y                         [vertical in image]

  Image mapping:
      u =  scale · x_proj + cx           [column]
      v = -scale · y_proj + cy           [row, negative because Y-up but v-down]

Tilt / perspective correction:
  If the tool's rotation axis is not perfectly vertical in the camera frame
  (e.g. due to mounting tilt), enable tilt correction.  The angle can be:
    - Provided manually with --tilt-angle
    - Auto-loaded from the master-mask metadata JSON (computed by
      build_master_masks_all_two_edge_tools.py) with --tilt-correction auto
  When active, every mask is de-rotated so the tool axis aligns with the
  image vertical before silhouette carving.

Output:
  Results are saved to --output-dir (default: 3D_reconstruction/output/<tool_id>/).
  Generated files:
    <tool_id>_visual_hull.obj        — Mesh (OBJ + optional STL via trimesh)
    <tool_id>_visual_hull.ply        — Point cloud (ASCII PLY)
    <tool_id>_visual_hull_voxels.npz — Raw voxel grid + bounds (numpy)
    <tool_id>_visual_hull_preview.png— 4-panel 3D/2D scatter preview
    <tool_id>_cross_sections.png     — Horizontal XZ slices along the axis
    run_config.json                  — Full record of parameters used

Usage examples:
    python visual_hull.py                                    # Defaults (tool002)
    python visual_hull.py --resolution 300                   # Higher voxel resolution
    python visual_hull.py --angle-mode uniform_360           # Uniform angles over 360°
    python visual_hull.py --angle-mode uniform_363           # 363 frames = 363 degrees
    python visual_hull.py --skip-views 3                     # Every 3rd view (faster)
    python visual_hull.py --flip-rotation                    # Reverse rotation direction
    python visual_hull.py --mask-dir "path/to/other_masks"   # Different tool
    python visual_hull.py --tilt-correction auto             # Auto-load tilt from metadata
    python visual_hull.py --tilt-correction 2.5              # Manual tilt angle in degrees
    python visual_hull.py --tilt-correction off              # No tilt (default)
"""

import numpy as np
import os
import sys
import glob
import re
import time
import json
import argparse
from pathlib import Path
from PIL import Image

# Optional imports (graceful fallback)
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ===========================================================================
# Default configuration
# ===========================================================================

# Paths are relative to THIS script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "DATA"))
DEFAULT_MASK_DIR = os.path.normpath(
    os.path.join(DATA_DIR, "masks", "tool002_final_masks")
)
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Master-mask metadata directory (for auto tilt correction)
DEFAULT_MASTER_MASK_META_DIR = os.path.normpath(
    os.path.join(
        DATA_DIR,
        "threshold_analysis",
        "master_mask_perspective",
        "tool_metadata",
    )
)

DEFAULT_ANGLE_MODE = "filename"   # 'filename', 'uniform_360', 'uniform_363'
DEFAULT_RESOLUTION = 200          # voxels per XZ axis
DEFAULT_SKIP_VIEWS = 1            # 1 = use all views
DEFAULT_MASK_SCALE = 0.5          # downsample masks to save memory on carving
DEFAULT_TOOL_ID = "tool002"
DEFAULT_TILT_CORRECTION = "off"   # 'off', 'auto', or a float angle in degrees


# ===========================================================================
# Mask discovery and angle parsing
# ===========================================================================

def discover_masks(mask_dir):
    """Find all mask image files in the directory.

    Returns:
        List of (filepath, filename) tuples, sorted by filename.
    """
    patterns = ["*.tiff", "*.tif", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(mask_dir, pat)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No mask images found in: {mask_dir}")
    return [(f, os.path.basename(f)) for f in files]


def parse_angles_from_filenames(file_list):
    """Extract rotation angles from filenames like '0045.50_degrees.tiff'.

    Returns:
        List of angles in degrees matching the file_list order.
    """
    angles = []
    for filepath, fname in file_list:
        match = re.match(r"(\d+\.\d+)_degrees\.tiff?$", fname, re.IGNORECASE)
        if match:
            angles.append(float(match.group(1)))
        else:
            raise ValueError(f"Cannot parse angle from filename: {fname}")
    return angles


def compute_uniform_angles(n_frames, total_degrees=360.0):
    """Create uniformly spaced angles over [0, total_degrees).

    Returns:
        List of n_frames angles: [0, step, 2*step, ..., (n-1)*step]
    """
    return [i * total_degrees / n_frames for i in range(n_frames)]


# ===========================================================================
# Tilt / perspective correction
# ===========================================================================

def resolve_tilt_angle(tilt_arg, tool_id, meta_dir=None):
    """Resolve the --tilt-correction argument into a numeric angle.

    Args:
        tilt_arg : str — 'off', 'auto', or a numeric string (degrees)
        tool_id  : str — e.g. 'tool002' (used for auto-lookup)
        meta_dir : str — directory containing <tool_id>_master_mask_metadata.json

    Returns:
        float — the counter-rotation angle in degrees (0.0 if off/not found)
    """
    if tilt_arg.lower() == "off":
        return 0.0

    # Try interpreting as a number first
    try:
        angle = float(tilt_arg)
        print(f"  Tilt correction : MANUAL = {angle:.4f}°")
        return angle
    except ValueError:
        pass

    if tilt_arg.lower() == "auto":
        if meta_dir is None:
            meta_dir = DEFAULT_MASTER_MASK_META_DIR
        json_path = os.path.join(meta_dir, f"{tool_id}_master_mask_metadata.json")
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            angle = meta.get("rotation_angle_deg", 0.0)
            status = meta.get("status", "unknown")
            print(f"  Tilt correction : AUTO from master-mask metadata")
            print(f"    Metadata file : {json_path}")
            print(f"    Tilt angle    : {meta.get('tilt_angle_deg', 0.0):.6f}°")
            print(f"    Counter-rot   : {angle:.6f}°  (status: {status})")
            return float(angle)
        else:
            print(f"  Tilt correction : AUTO requested but metadata not found:")
            print(f"    {json_path}")
            print(f"  → Falling back to 0° (no correction)")
            return 0.0

    print(f"  [WARN] Unrecognised --tilt-correction value: '{tilt_arg}' — using 0°")
    return 0.0


def apply_tilt_correction(mask_array, tilt_angle_deg):
    """Rotate a binary mask by the given angle to correct axis tilt.

    Uses PIL rotation (bilinear, expand=False) and re-thresholds to binary.

    Args:
        mask_array     : 2D uint8 numpy array (0/255)
        tilt_angle_deg : counter-rotation angle in degrees

    Returns:
        Corrected 2D uint8 numpy array (0/255)
    """
    if abs(tilt_angle_deg) < 1e-6:
        return mask_array
    pil_img = Image.fromarray(mask_array)
    # PIL.rotate: positive = counter-clockwise — that matches our convention
    rotated = pil_img.rotate(tilt_angle_deg, resample=Image.BILINEAR,
                             expand=False, fillcolor=0)
    arr = np.array(rotated)
    return (arr > 127).astype(np.uint8) * 255


# ===========================================================================
# Silhouette analysis — auto-estimate projection parameters
# ===========================================================================

def analyze_silhouettes(file_list, sample_count=30, tilt_angle_deg=0.0):
    """Analyze a sample of silhouettes to estimate projection parameters.

    Scans a subset of masks to find:
      - cx, cy : rotation axis position in image pixels
      - scale  : pixels per normalised world unit
      - y_min, y_max : vertical extent of the tool in world units
      - image_width, image_height

    The world coordinate system is set so that the tool's maximum
    cross-sectional radius equals 1.0 world units.

    Args:
        file_list       : list of (filepath, filename) tuples
        sample_count    : how many views to sample
        tilt_angle_deg  : if non-zero, de-rotate masks before analysis

    Returns:
        dict with the above keys plus diagnostics.
    """
    n = len(file_list)
    indices = np.linspace(0, n - 1, min(sample_count, n), dtype=int)

    h_centers = []      # horizontal centre of each silhouette
    v_centers = []      # vertical centre
    h_half_widths = []  # half the horizontal extent
    v_tops = []         # top-most row with mask pixels
    v_bottoms = []      # bottom-most row
    img_w = img_h = None

    print(f"  Sampling {len(indices)} silhouettes for parameter estimation ...")

    for idx in indices:
        filepath, _ = file_list[idx]
        img = np.array(Image.open(filepath))

        # Apply tilt correction if requested
        if abs(tilt_angle_deg) > 1e-6:
            img = apply_tilt_correction(img, tilt_angle_deg)

        mask = img > 127

        if img_w is None:
            img_h, img_w = mask.shape

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows):
            continue

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h_centers.append((cmin + cmax) / 2.0)
        v_centers.append((rmin + rmax) / 2.0)
        h_half_widths.append((cmax - cmin) / 2.0)
        v_tops.append(rmin)
        v_bottoms.append(rmax)

    # --- Rotation axis horizontal position (should be consistent) ---
    cx = float(np.median(h_centers))

    # --- Vertical centre (constant across rotation) ---
    cy = float(np.median(v_centers))

    # --- Maximum silhouette half-width → sets the "1 unit" radius ---
    max_half_width = float(np.max(h_half_widths))
    max_radius_world = 1.0
    scale = max_half_width / max_radius_world   # pixels per world unit

    # --- Vertical extent in world units ---
    # Image mapping: v = -scale * Y + cy  =>  Y = (cy - v) / scale
    y_top    = (cy - float(np.min(v_tops)))    / scale   # highest row → max Y
    y_bottom = (cy - float(np.max(v_bottoms))) / scale   # lowest row  → min Y

    # Padding (10 %)
    y_pad = 0.1 * (y_top - y_bottom)
    y_min = y_bottom - y_pad
    y_max = y_top + y_pad

    result = {
        "cx": cx,
        "cy": cy,
        "scale": scale,
        "y_min": y_min,
        "y_max": y_max,
        "max_radius": max_radius_world,
        "image_width": img_w,
        "image_height": img_h,
        "max_half_width_px": max_half_width,
    }

    print(f"    Image size        : {img_w} × {img_h}")
    print(f"    Rotation axis (cx): {cx:.1f} px")
    print(f"    Vertical centre   : {cy:.1f} px")
    print(f"    Max half-width    : {max_half_width:.1f} px  →  1.0 world units")
    print(f"    Scale             : {scale:.1f} px / world unit")
    print(f"    Volume Y range    : [{y_min:.3f}, {y_max:.3f}] world units")

    return result


# ===========================================================================
# Visual hull — voxel carving
# ===========================================================================

def visual_hull_carving(
    file_list,
    angles_deg,
    params,
    resolution=200,
    skip_views=1,
    mask_scale=1.0,
    flip_rotation=False,
    tilt_angle_deg=0.0,
):
    """Perform voxel carving to compute the visual hull.

    Loads masks one at a time to keep memory low.

    Args:
        file_list      : list of (filepath, filename) tuples
        angles_deg     : rotation angle for each frame (degrees)
        params         : dict from analyze_silhouettes()
        resolution     : voxels per XZ axis
        skip_views     : use every N-th view (1 = all)
        mask_scale     : downscale masks by this factor (0.5 = half resolution)
        flip_rotation  : negate rotation angles (try this if result looks wrong)
        tilt_angle_deg : if non-zero, de-rotate masks before carving

    Returns:
        voxel_grid    : 3D boolean array (True = occupied)
        volume_bounds : ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        grid_shape    : (nx, ny, nz)
    """
    cx = params["cx"]
    cy = params["cy"]
    scale = params["scale"]
    y_min = params["y_min"]
    y_max = params["y_max"]
    R = params["max_radius"]

    # Apply mask scaling to projection parameters
    cx_s = cx * mask_scale
    cy_s = cy * mask_scale
    scale_s = scale * mask_scale

    # Volume bounds with 15 % lateral padding
    pad = R * 0.15
    volume_bounds = (
        (-R - pad, R + pad),   # X
        (y_min, y_max),        # Y
        (-R - pad, R + pad),   # Z
    )
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = volume_bounds

    # Grid dimensions — scale ny to preserve aspect ratio
    nx = nz = resolution
    y_extent = ymax - ymin
    xz_extent = xmax - xmin
    ny = max(int(resolution * y_extent / xz_extent), 10)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)

    print(f"  Voxel grid : {nx} × {ny} × {nz} = {nx * ny * nz:,} voxels")
    print(f"  Volume     : X[{xmin:.3f}, {xmax:.3f}]  "
          f"Y[{ymin:.3f}, {ymax:.3f}]  Z[{zmin:.3f}, {zmax:.3f}]")

    # Pre-compute voxel coordinates (flat)
    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing="ij")  # shape (nx, ny, nz)
    X_flat = X3.ravel().astype(np.float32)
    Y_flat = Y3.ravel().astype(np.float32)
    Z_flat = Z3.ravel().astype(np.float32)
    del X3, Y3, Z3  # free memory

    occupied = np.ones(X_flat.size, dtype=bool)

    # Select views
    view_indices = list(range(0, len(file_list), skip_views))
    n_views = len(view_indices)

    print(f"  Views      : {n_views} (every {skip_views}-th frame)")
    print(f"  Mask scale : {mask_scale}")
    print()

    t_start = time.time()

    for vi, idx in enumerate(view_indices):
        filepath, fname = file_list[idx]
        theta_deg = angles_deg[idx]
        if flip_rotation:
            theta_deg = -theta_deg
        theta = np.radians(theta_deg)

        cos_t = np.cos(theta).astype(np.float32)
        sin_t = np.sin(theta).astype(np.float32)

        # Load mask
        img = np.array(Image.open(filepath))

        # Apply tilt correction if requested
        if abs(tilt_angle_deg) > 1e-6:
            img = apply_tilt_correction(img, tilt_angle_deg)

        mask = img > 127

        if mask_scale != 1.0:
            new_w = int(mask.shape[1] * mask_scale)
            new_h = int(mask.shape[0] * mask_scale)
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (new_w, new_h), Image.NEAREST
                )
            ) > 127

        h_img, w_img = mask.shape

        # --- Project only currently-occupied voxels (huge speedup) ---
        occ_idx = np.where(occupied)[0]
        if len(occ_idx) == 0:
            print("  All voxels carved — stopping early.")
            break

        # Orthographic projection at angle θ
        x_proj = X_flat[occ_idx] * cos_t + Z_flat[occ_idx] * sin_t
        y_proj = Y_flat[occ_idx]

        # Map to (scaled) image coordinates
        u = (scale_s * x_proj + cx_s).astype(np.int32)
        v = (-scale_s * y_proj + cy_s).astype(np.int32)

        # Bounds check
        in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)

        # Silhouette test
        in_silhouette = np.zeros(len(occ_idx), dtype=bool)
        ib = np.where(in_bounds)[0]
        if len(ib) > 0:
            in_silhouette[ib] = mask[v[ib], u[ib]]

        # Carve: voxels inside image but outside silhouette → remove
        carved = in_bounds & ~in_silhouette
        occupied[occ_idx[carved]] = False

        # Progress
        if (vi + 1) % 50 == 0 or vi == 0 or (vi + 1) == n_views:
            elapsed = time.time() - t_start
            rate = (vi + 1) / elapsed if elapsed > 0 else 1
            remaining = (n_views - vi - 1) / rate
            print(
                f"  View {vi + 1:4d}/{n_views} ({theta_deg:7.2f}°) : "
                f"{np.sum(occupied):>10,} voxels remaining  "
                f"[{elapsed:.1f}s elapsed, ~{remaining:.0f}s left]"
            )

    elapsed = time.time() - t_start
    n_occ = int(np.sum(occupied))
    print(f"\n  Carving complete in {elapsed:.1f}s")
    print(f"  Occupied : {n_occ:,} / {occupied.size:,} "
          f"({100 * n_occ / occupied.size:.2f}%)")

    voxel_grid = occupied.reshape(nx, ny, nz)
    return voxel_grid, volume_bounds, (nx, ny, nz)


# ===========================================================================
# Export — mesh, point cloud, voxel grid
# ===========================================================================

def export_mesh(voxel_grid, volume_bounds, output_path, smooth_sigma=0.5):
    """Extract mesh via marching cubes and save as OBJ.

    Optionally applies light Gaussian smoothing to the voxel grid before
    marching cubes for a cleaner surface.
    """
    if not HAS_SKIMAGE:
        print("  [SKIP] scikit-image not installed — cannot extract mesh.")
        print("         Install with:  pip install scikit-image")
        return None

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = volume_bounds
    nx, ny, nz = voxel_grid.shape

    # Pad with zeros to guarantee a closed surface
    volume = np.pad(voxel_grid.astype(np.float32), 1, mode="constant",
                    constant_values=0)

    if HAS_SCIPY and smooth_sigma > 0:
        volume = gaussian_filter(volume, sigma=smooth_sigma)

    try:
        verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5)
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None

    # Remove padding offset and scale to world coordinates
    verts -= 1.0
    verts[:, 0] = verts[:, 0] / max(nx - 1, 1) * (xmax - xmin) + xmin
    verts[:, 1] = verts[:, 1] / max(ny - 1, 1) * (ymax - ymin) + ymin
    verts[:, 2] = verts[:, 2] / max(nz - 1, 1) * (zmax - zmin) + zmin

    # Ensure .obj extension
    output_path = str(output_path)
    base, ext = os.path.splitext(output_path)
    if ext.lower() not in (".obj", ".stl", ".ply"):
        output_path = base + ".obj"

    if HAS_TRIMESH:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                               vertex_normals=normals)
        mesh.export(output_path)
        # Also save the other common format
        other_ext = ".stl" if output_path.endswith(".obj") else ".obj"
        mesh.export(base + other_ext)
        print(f"  Mesh saved : {output_path}  (also {base + other_ext})")
    else:
        # Manual OBJ export (no external dependency)
        if not output_path.endswith(".obj"):
            output_path = base + ".obj"
        with open(output_path, "w") as f:
            f.write(f"# Visual Hull mesh — {len(verts)} vertices, {len(faces)} faces\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in faces:
                # OBJ is 1-indexed;  v//vn format
                f.write(f"f {face[0]+1}//{face[0]+1} "
                        f"{face[1]+1}//{face[1]+1} "
                        f"{face[2]+1}//{face[2]+1}\n")
        print(f"  Mesh saved (OBJ): {output_path}")

    print(f"    Vertices : {len(verts):,}   Faces : {len(faces):,}")
    return verts, faces


def export_point_cloud(voxel_grid, volume_bounds, output_path):
    """Export occupied voxel centres as a PLY point cloud."""
    nx, ny, nz = voxel_grid.shape
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = volume_bounds

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    occ = voxel_grid.ravel().astype(bool)
    pts = np.column_stack([X.ravel()[occ], Y.ravel()[occ], Z.ravel()[occ]])

    output_path = str(output_path)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    print(f"  Point cloud saved : {output_path}  ({len(pts):,} points)")


def save_voxel_grid(voxel_grid, volume_bounds, output_path):
    """Save voxel grid as compressed numpy archive."""
    np.savez_compressed(
        str(output_path),
        voxel_grid=voxel_grid,
        volume_bounds=np.array(volume_bounds),
    )
    print(f"  Voxel grid saved  : {output_path}")


# ===========================================================================
# Visualisation
# ===========================================================================

def visualize_result(voxel_grid, volume_bounds, output_dir, tool_id,
                     max_display=50000):
    """Create a 3D scatter plot of the visual hull and save as PNG."""
    if not HAS_MATPLOTLIB:
        print("  [SKIP] matplotlib not installed — cannot create preview.")
        return

    nx, ny, nz = voxel_grid.shape
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = volume_bounds

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    occ = voxel_grid.ravel().astype(bool)
    pts_x = X.ravel()[occ]
    pts_y = Y.ravel()[occ]
    pts_z = Z.ravel()[occ]

    n_pts = len(pts_x)
    if n_pts == 0:
        print("  No occupied voxels — nothing to visualise.")
        return

    # Subsample for manageable rendering
    if n_pts > max_display:
        idx = np.random.default_rng(42).choice(n_pts, max_display, replace=False)
        pts_x, pts_y, pts_z = pts_x[idx], pts_y[idx], pts_z[idx]

    fig = plt.figure(figsize=(14, 10))

    # --- View 1: 3D scatter ---
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(pts_x, pts_z, pts_y, s=0.3, alpha=0.2, c=pts_y, cmap="viridis")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y (axis)")
    ax1.set_title("3D view")
    _set_equal_aspect_3d(ax1, volume_bounds)

    # --- View 2: Top-down (XZ plane) ---
    ax2 = fig.add_subplot(222)
    ax2.scatter(pts_x, pts_z, s=0.1, alpha=0.1, c="steelblue")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Top-down (XZ)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # --- View 3: Front view (XY plane) ---
    ax3 = fig.add_subplot(223)
    ax3.scatter(pts_x, pts_y, s=0.1, alpha=0.1, c="steelblue")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y (axis)")
    ax3.set_title("Front (XY)")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # --- View 4: Side view (ZY plane) ---
    ax4 = fig.add_subplot(224)
    ax4.scatter(pts_z, pts_y, s=0.1, alpha=0.1, c="steelblue")
    ax4.set_xlabel("Z")
    ax4.set_ylabel("Y (axis)")
    ax4.set_title("Side (ZY)")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"Visual Hull — {tool_id}  ({n_pts:,} voxels, showing {len(pts_x):,})",
        fontsize=13,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{tool_id}_visual_hull_preview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Preview image saved : {save_path}")
    plt.close(fig)


def _set_equal_aspect_3d(ax, volume_bounds):
    """Set equal aspect ratio on a 3D axes."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = volume_bounds
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2
    mid_z = (zmin + zmax) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)


# ===========================================================================
# Cross-section slicing (bonus diagnostic)
# ===========================================================================

def save_cross_sections(voxel_grid, volume_bounds, output_dir, tool_id,
                        n_slices=5):
    """Save horizontal cross-section images along the Y axis."""
    if not HAS_MATPLOTLIB:
        return

    nx, ny, nz = voxel_grid.shape
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = volume_bounds

    slice_indices = np.linspace(0, ny - 1, n_slices + 2, dtype=int)[1:-1]

    fig, axes = plt.subplots(1, len(slice_indices), figsize=(4 * len(slice_indices), 4))
    if len(slice_indices) == 1:
        axes = [axes]

    for ax, yi in zip(axes, slice_indices):
        y_val = ymin + yi / max(ny - 1, 1) * (ymax - ymin)
        section = voxel_grid[:, yi, :]  # XZ slice
        ax.imshow(section.T, origin="lower", cmap="gray",
                  extent=[xmin, xmax, zmin, zmax], aspect="equal")
        ax.set_title(f"Y = {y_val:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    fig.suptitle(f"Horizontal cross-sections — {tool_id}", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{tool_id}_cross_sections.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Cross-sections saved : {save_path}")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visual Hull reconstruction from turntable silhouettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mask-dir", type=str, default=DEFAULT_MASK_DIR,
        help="Path to directory containing mask images",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Path to output directory.  Default: 3D_reconstruction/output/<tool_id>/"
        ),
    )
    parser.add_argument(
        "--angle-mode", type=str, default=DEFAULT_ANGLE_MODE,
        choices=["filename", "uniform_360", "uniform_363"],
        help=(
            "How to determine rotation angles:\n"
            "  filename    — parse from filenames (default)\n"
            "  uniform_360 — N frames uniformly over 360°\n"
            "  uniform_363 — N frames uniformly over 363°"
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION,
        help="Voxels per XZ axis (default: 200)",
    )
    parser.add_argument(
        "--skip-views", type=int, default=DEFAULT_SKIP_VIEWS,
        help="Use every N-th view for faster computation (default: 1 = all)",
    )
    parser.add_argument(
        "--mask-scale", type=float, default=DEFAULT_MASK_SCALE,
        help="Down-scale factor for masks during carving (default: 0.5)",
    )
    parser.add_argument(
        "--flip-rotation", action="store_true",
        help="Negate all rotation angles (try this if result looks wrong)",
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=0.5,
        help="Gaussian sigma for mesh smoothing (0 = no smoothing)",
    )
    parser.add_argument(
        "--no-mesh", action="store_true", help="Skip mesh export",
    )
    parser.add_argument(
        "--no-pointcloud", action="store_true", help="Skip point cloud export",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualisation",
    )
    parser.add_argument(
        "--tool-id", type=str, default=DEFAULT_TOOL_ID,
        help="Tool identifier for output file naming (default: tool002)",
    )
    parser.add_argument(
        "--tilt-correction", type=str, default=DEFAULT_TILT_CORRECTION,
        metavar="MODE",
        help=(
            "Perspective / tilt correction for the rotation axis:\n"
            "  off   — no correction (default)\n"
            "  auto  — load counter-rotation angle from master-mask metadata\n"
            "  <N>   — manual angle in degrees (e.g. 2.5)"
        ),
    )
    parser.add_argument(
        "--tilt-meta-dir", type=str, default=None,
        help=(
            "Directory containing <tool_id>_master_mask_metadata.json files. "
            "Used only when --tilt-correction auto.  Default: "
            "DATA/threshold_analysis/master_mask_perspective/tool_metadata/"
        ),
    )

    args = parser.parse_args()

    # Resolve output directory — default to output/<tool_id>/
    if args.output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.tool_id)
    else:
        output_dir = os.path.abspath(args.output_dir)

    mask_dir = os.path.abspath(args.mask_dir)
    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{args.tool_id}_visual_hull"

    # ------------------------------------------------------------------
    # Resolve tilt correction
    # ------------------------------------------------------------------
    tilt_angle = resolve_tilt_angle(
        args.tilt_correction,
        args.tool_id,
        meta_dir=args.tilt_meta_dir,
    )
    tilt_active = abs(tilt_angle) > 1e-6

    # Banner
    print("=" * 72)
    print("  Visual Hull (Shape from Silhouette) Reconstruction")
    print("=" * 72)
    print(f"  Mask directory   : {mask_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Angle mode       : {args.angle_mode}")
    print(f"  Resolution       : {args.resolution}")
    print(f"  Skip views       : {args.skip_views}")
    print(f"  Mask scale       : {args.mask_scale}")
    print(f"  Flip rotation    : {args.flip_rotation}")
    print(f"  Smooth sigma     : {args.smooth_sigma}")
    print(f"  Tool ID          : {args.tool_id}")
    print(f"  Tilt correction  : {'ON (' + f'{tilt_angle:.4f}°)' if tilt_active else 'OFF'}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Discover masks
    # ------------------------------------------------------------------
    print("Step 1 — Discovering masks ...")
    file_list = discover_masks(mask_dir)
    print(f"  Found {len(file_list)} mask files\n")

    # ------------------------------------------------------------------
    # Step 2: Determine rotation angles
    # ------------------------------------------------------------------
    print("Step 2 — Determining rotation angles ...")
    if args.angle_mode == "filename":
        angles_deg = parse_angles_from_filenames(file_list)
        print(f"  Parsed from filenames: {angles_deg[0]:.2f}° → "
              f"{angles_deg[-1]:.2f}°  (Δ ≈ {np.median(np.diff(angles_deg)):.3f}°)")
    elif args.angle_mode == "uniform_360":
        angles_deg = compute_uniform_angles(len(file_list), 360.0)
        print(f"  Uniform over 360°: step = {360.0 / len(file_list):.4f}°")
    elif args.angle_mode == "uniform_363":
        angles_deg = compute_uniform_angles(len(file_list), 363.0)
        print(f"  Uniform over 363°: step = {363.0 / len(file_list):.4f}°")
    print()

    # ------------------------------------------------------------------
    # Step 3: Analyse silhouettes
    # ------------------------------------------------------------------
    print("Step 3 — Analysing silhouette extents ...")
    params = analyze_silhouettes(file_list, tilt_angle_deg=tilt_angle)
    print()

    # ------------------------------------------------------------------
    # Step 4: Voxel carving
    # ------------------------------------------------------------------
    print("Step 4 — Voxel carving ...")
    voxel_grid, volume_bounds, grid_shape = visual_hull_carving(
        file_list,
        angles_deg,
        params,
        resolution=args.resolution,
        skip_views=args.skip_views,
        mask_scale=args.mask_scale,
        flip_rotation=args.flip_rotation,
        tilt_angle_deg=tilt_angle,
    )
    print()

    # ------------------------------------------------------------------
    # Step 5: Export
    # ------------------------------------------------------------------
    print("Step 5 — Exporting results ...")

    # Always save the raw voxel grid
    voxel_path = os.path.join(output_dir, f"{prefix}_voxels.npz")
    save_voxel_grid(voxel_grid, volume_bounds, voxel_path)

    # Mesh
    if not args.no_mesh:
        mesh_path = os.path.join(output_dir, f"{prefix}.obj")
        export_mesh(voxel_grid, volume_bounds, mesh_path,
                    smooth_sigma=args.smooth_sigma)

    # Point cloud
    if not args.no_pointcloud:
        pc_path = os.path.join(output_dir, f"{prefix}.ply")
        export_point_cloud(voxel_grid, volume_bounds, pc_path)

    # ------------------------------------------------------------------
    # Save run configuration (reproducibility)
    # ------------------------------------------------------------------
    from datetime import datetime
    run_config = {
        "tool_id": args.tool_id,
        "mask_dir": mask_dir,
        "output_dir": output_dir,
        "angle_mode": args.angle_mode,
        "num_masks": len(file_list),
        "angle_range": [float(angles_deg[0]), float(angles_deg[-1])],
        "resolution": args.resolution,
        "skip_views": args.skip_views,
        "mask_scale": args.mask_scale,
        "flip_rotation": args.flip_rotation,
        "smooth_sigma": args.smooth_sigma,
        "tilt_correction": args.tilt_correction,
        "tilt_angle_applied_deg": tilt_angle,
        "grid_shape": list(grid_shape),
        "volume_bounds": [list(b) for b in volume_bounds],
        "occupied_voxels": int(np.sum(voxel_grid)),
        "total_voxels": int(np.prod(grid_shape)),
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Run config saved  : {config_path}")
    print()

    # ------------------------------------------------------------------
    # Step 6: Visualisation
    # ------------------------------------------------------------------
    if not args.no_viz:
        print("Step 6 — Visualisation ...")
        visualize_result(voxel_grid, volume_bounds, output_dir, args.tool_id)
        save_cross_sections(voxel_grid, volume_bounds, output_dir, args.tool_id)

    print()
    print("=" * 72)
    print("  Done!  Outputs in:", output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
