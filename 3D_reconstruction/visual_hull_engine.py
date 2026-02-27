#!/usr/bin/env python3
"""
Visual Hull Engine — Optimized core for Shape from Silhouette reconstruction.
==============================================================================

Hardware-optimized for:
  - Intel Iris XE GPU   → NumPy vectorised ops (no CUDA dependency)
  - 8P + 8E cores (16T) → parallel mask I/O & tilt correction via ThreadPool
  - 16 GB RAM           → chunked carving, float32, in-place boolean ops

Public API (used by the GUI and CLI):
    discover_tools()          → list of tool_ids with mask folders
    EngineConfig              → dataclass with all parameters
    run_visual_hull(config, progress_callback=None)  → results dict

All heavy functions accept an optional ``progress_callback(percent, message)``
so the GUI can update a progress bar without polling.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from skimage import measure as sk_measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for threads)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "DATA"))
MASKS_DIR   = os.path.join(DATA_DIR, "masks")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")

MASTER_MASK_META_DIR = os.path.normpath(
    os.path.join(DATA_DIR, "threshold_analysis",
                 "master_mask_perspective", "tool_metadata")
)

MESHLAB_EXE = r"C:\Program Files\VCG\MeshLab\meshlab.exe"

# Hardware
MAX_WORKERS = 12   # ≤ 16 logical cores; leaves 4 for OS + GUI


# ═══════════════════════════════════════════════════════════════════════════
#  OpenCL / GPU helpers  (Intel Iris Xe optimised)
# ═══════════════════════════════════════════════════════════════════════════

# OpenCL kernel — one work-item per voxel, processes ONE view at a time.
# The `occupied` buffer stays on the GPU across all views (zero round-trips).
# Only the flattened mask is uploaded per view (~1-3 MB at 0.5 scale).
_OPENCL_KERNEL_SRC = r"""
__kernel void carve(
    __global const float *X,
    __global const float *Y,
    __global const float *Z,
    __global uchar       *occupied,
    __global const uchar *mask,
    const float cos_t,
    const float sin_t,
    const float cx,
    const float cy,
    const float scale,
    const int   mask_w,
    const int   mask_h,
    const int   n_voxels)
{
    int i = get_global_id(0);
    if (i >= n_voxels) return;
    if (occupied[i] == 0) return;          // already carved

    float xp = X[i] * cos_t + Z[i] * sin_t;
    float yp = Y[i];

    int u = (int)(scale * xp + cx);
    int v = (int)(-scale * yp + cy);

    // Inside image bounds?
    if (u >= 0 && u < mask_w && v >= 0 && v < mask_h) {
        if (mask[v * mask_w + u] == 0) {
            occupied[i] = 0;              // carve!
        }
    }
    // Voxels projecting outside the image are NOT carved (conservative)
}
"""


def _get_opencl_gpu_context():
    """Try to create an OpenCL context on the Intel Iris Xe GPU.

    Returns (context, device_name) or (None, reason_string).
    """
    if not HAS_OPENCL:
        return None, "pyopencl not installed"
    try:
        platforms = cl.get_platforms()
        for plat in platforms:
            for dev in plat.get_devices(device_type=cl.device_type.GPU):
                ctx = cl.Context([dev])
                return ctx, dev.name.strip()
        # No GPU found, try any accelerator
        for plat in platforms:
            for dev in plat.get_devices():
                if dev.type != cl.device_type.CPU:
                    ctx = cl.Context([dev])
                    return ctx, dev.name.strip()
        return None, "No GPU device found in OpenCL platforms"
    except Exception as e:
        return None, str(e)


def get_gpu_info() -> dict:
    """Return GPU availability info (used by the GUI)."""
    ctx, info = _get_opencl_gpu_context()
    if ctx is not None:
        dev = ctx.devices[0]
        return {
            "available": True,
            "device": info,
            "mem_mb": dev.global_mem_size // (1024 * 1024),
            "compute_units": dev.max_compute_units,
            "max_work_group": dev.max_work_group_size,
        }
    return {"available": False, "device": info}


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EngineConfig:
    """All parameters for one visual hull run."""
    tool_id:          str   = "tool002"
    mask_dir:         str   = ""          # auto-resolved if empty
    output_dir:       str   = ""          # auto-resolved if empty
    angle_mode:       str   = "filename"  # filename | uniform_360 | uniform_363
    resolution:       int   = 200         # voxels per XZ axis
    skip_views:       int   = 1           # 1 = all views
    mask_scale:       float = 0.5         # downscale factor
    flip_rotation:    bool  = False
    smooth_sigma:     float = 0.5         # Gaussian sigma for mesh
    tilt_correction:  str   = "off"       # off | auto | <float>
    tilt_meta_dir:    str   = ""          # auto-resolved if empty
    symmetry_half:    bool  = False      # use first half of images (180°) for symmetric tools
    export_mesh:      bool  = True
    export_pointcloud:bool  = True
    export_viz:       bool  = True
    use_gpu:          bool  = True       # try OpenCL GPU; auto-fallback to CPU
    n_workers:        int   = MAX_WORKERS

    def resolve(self):
        """Fill in auto-computed paths."""
        if not self.mask_dir:
            self.mask_dir = os.path.join(MASKS_DIR,
                                         f"{self.tool_id}_final_masks")
        if not self.output_dir:
            self.output_dir = os.path.join(OUTPUT_ROOT, self.tool_id)
        if not self.tilt_meta_dir:
            self.tilt_meta_dir = MASTER_MASK_META_DIR
        self.mask_dir   = os.path.normpath(self.mask_dir)
        self.output_dir = os.path.normpath(self.output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Tool discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_tools() -> List[str]:
    """Return sorted list of tool_ids that have a *_final_masks folder."""
    if not os.path.isdir(MASKS_DIR):
        return []
    tool_ids = []
    for name in sorted(os.listdir(MASKS_DIR)):
        m = re.match(r"(tool\d+)_final_masks$", name)
        if m and os.path.isdir(os.path.join(MASKS_DIR, name)):
            tool_ids.append(m.group(1))
    return tool_ids


# ═══════════════════════════════════════════════════════════════════════════
#  Mask I/O helpers  (thread-safe — PIL per file, no shared state)
# ═══════════════════════════════════════════════════════════════════════════

def _discover_mask_files(mask_dir: str) -> List[Tuple[str, str]]:
    """Return sorted list of (filepath, filename) in mask_dir."""
    files = []
    for ext in ("*.tiff", "*.tif", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(mask_dir, ext)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No mask images in: {mask_dir}")
    return [(f, os.path.basename(f)) for f in files]


def _parse_angles_from_filenames(file_list):
    angles = []
    for _, fname in file_list:
        m = re.match(r"(\d+\.\d+)_degrees\.tiff?$", fname, re.IGNORECASE)
        if m:
            angles.append(float(m.group(1)))
        else:
            raise ValueError(f"Cannot parse angle from: {fname}")
    return np.array(angles, dtype=np.float64)


def _uniform_angles(n, total=360.0):
    return np.linspace(0, total, n, endpoint=False)


def _load_mask(filepath: str, scale: float, tilt_deg: float) -> np.ndarray:
    """Load a mask, optionally tilt-correct and downscale.  Returns bool array."""
    img = np.array(Image.open(filepath))
    if abs(tilt_deg) > 1e-6:
        pil = Image.fromarray(img)
        pil = pil.rotate(tilt_deg, resample=Image.BILINEAR,
                          expand=False, fillcolor=0)
        img = np.array(pil)
    mask = img > 127
    if scale != 1.0:
        h, w = mask.shape
        new_w, new_h = int(w * scale), int(h * scale)
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (new_w, new_h), Image.NEAREST)
        ) > 127
    return mask


# ═══════════════════════════════════════════════════════════════════════════
#  Tilt angle resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_tilt_angle(tilt_arg: str, tool_id: str, meta_dir: str) -> float:
    """Return counter-rotation angle in degrees (0 if off)."""
    if tilt_arg.lower() == "off":
        return 0.0
    try:
        return float(tilt_arg)
    except ValueError:
        pass
    if tilt_arg.lower() == "auto":
        jpath = os.path.join(meta_dir,
                             f"{tool_id}_master_mask_metadata.json")
        if os.path.isfile(jpath):
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return float(meta.get("rotation_angle_deg", 0.0))
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Silhouette analysis
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_silhouettes(file_list, tilt_deg, mask_scale, n_workers,
                         sample_n=30):
    """Estimate projection params from a sample of masks (parallel load)."""
    n = len(file_list)
    indices = np.linspace(0, n - 1, min(sample_n, n), dtype=int)
    sample_files = [file_list[i][0] for i in indices]

    # Parallel load
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        masks = list(pool.map(
            lambda fp: _load_mask(fp, mask_scale, tilt_deg),
            sample_files
        ))

    h_centers, h_halfs, v_tops, v_bots = [], [], [], []
    img_h, img_w = masks[0].shape

    for mask in masks:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows):
            continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h_centers.append((cmin + cmax) / 2.0)
        h_halfs.append((cmax - cmin) / 2.0)
        v_tops.append(rmin)
        v_bots.append(rmax)

    cx = float(np.median(h_centers))
    cy = float(np.median([(t + b) / 2 for t, b in zip(v_tops, v_bots)]))
    max_hw = float(np.max(h_halfs))
    scale = max_hw  # pixels per 1.0 world unit (max radius)

    y_top    = (cy - float(np.min(v_tops)))  / scale
    y_bottom = (cy - float(np.max(v_bots)))  / scale
    y_pad = 0.10 * (y_top - y_bottom)

    return {
        "cx": cx, "cy": cy, "scale": scale,
        "y_min": y_bottom - y_pad, "y_max": y_top + y_pad,
        "max_radius": 1.0,
        "img_w": img_w, "img_h": img_h,
        "max_half_width_px": max_hw,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Voxel carving — GPU path  (Intel Iris Xe via OpenCL)
# ═══════════════════════════════════════════════════════════════════════════

def _carve_gpu(
    file_list, angles_deg, params, cfg: EngineConfig, tilt_deg: float,
    progress_cb: Optional[Callable] = None,
):
    """GPU-accelerated voxel carving via OpenCL.

    - Voxel coords + occupied array live on the GPU for the ENTIRE run.
    - Per view: only the mask image (~1-3 MB) is uploaded.
    - Mask pre-loading still uses the CPU thread pool (overlaps with GPU work).
    """
    ctx, dev_name = _get_opencl_gpu_context()
    if ctx is None:
        raise RuntimeError(f"GPU not available: {dev_name}")

    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, _OPENCL_KERNEL_SRC).build()
    mf = cl.mem_flags

    cx_s = params["cx"]
    cy_s = params["cy"]
    sc_s = params["scale"]
    R    = params["max_radius"]
    ymin = params["y_min"]
    ymax = params["y_max"]

    pad = R * 0.15
    bounds = ((-R - pad, R + pad), (ymin, ymax), (-R - pad, R + pad))
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds

    nx = nz = cfg.resolution
    ny = max(int(cfg.resolution * (yhi - ylo) / (xhi - xlo)), 10)

    x = np.linspace(xlo, xhi, nx, dtype=np.float32)
    y = np.linspace(ylo, yhi, ny, dtype=np.float32)
    z = np.linspace(zlo, zhi, nz, dtype=np.float32)

    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    Xf = Xg.ravel().astype(np.float32)
    Yf = Yg.ravel().astype(np.float32)
    Zf = Zg.ravel().astype(np.float32)
    del Xg, Yg, Zg

    n_voxels = Xf.size
    occupied_host = np.ones(n_voxels, dtype=np.uint8)

    # ---- Upload voxel data to GPU (stays there the entire run) ----
    buf_X   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=Xf)
    buf_Y   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=Yf)
    buf_Z   = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=Zf)
    buf_occ = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=occupied_host)
    del Xf, Yf, Zf  # free CPU copies

    # Work-group size: round up to 256 (good for Iris Xe EU topology)
    local_size = 256
    global_size = ((n_voxels + local_size - 1) // local_size) * local_size

    view_idx = list(range(0, len(file_list), cfg.skip_views))
    n_views = len(view_idx)

    # Prefetch masks on CPU threads while GPU is busy
    PREFETCH = min(cfg.n_workers * 2, n_views)
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, Future

    pool = ThreadPoolExecutor(max_workers=cfg.n_workers)
    prefetch_q: deque = deque()

    def submit_load(vi_pos):
        idx = view_idx[vi_pos]
        fp = file_list[idx][0]
        return pool.submit(_load_mask, fp, cfg.mask_scale, tilt_deg)

    for i in range(min(PREFETCH, n_views)):
        prefetch_q.append((i, submit_load(i)))

    # Pre-allocate a mask buffer on GPU (reuse across views)
    # We'll create it after loading the first mask to know the size.
    buf_mask = None
    mask_buf_size = 0

    t0 = time.time()

    for vi in range(n_views):
        qi, fut = prefetch_q.popleft()
        assert qi == vi
        mask = fut.result()

        next_i = vi + PREFETCH
        if next_i < n_views:
            prefetch_q.append((next_i, submit_load(next_i)))

        idx = view_idx[vi]
        theta_deg = angles_deg[idx]
        if cfg.flip_rotation:
            theta_deg = -theta_deg
        theta = np.radians(theta_deg)

        h_img, w_img = mask.shape
        mask_flat = mask.astype(np.uint8).ravel()  # 0 or 1 (bool→uint8)

        # Create / reuse mask buffer on GPU
        needed = mask_flat.nbytes
        if buf_mask is None or needed != mask_buf_size:
            if buf_mask is not None:
                buf_mask.release()
            buf_mask = cl.Buffer(ctx, mf.READ_ONLY, size=needed)
            mask_buf_size = needed

        # Upload mask (async — overlaps with previous kernel if still running)
        cl.enqueue_copy(queue, buf_mask, mask_flat, is_blocking=False)

        # Launch kernel
        prg.carve(
            queue, (global_size,), (local_size,),
            buf_X, buf_Y, buf_Z, buf_occ, buf_mask,
            np.float32(np.cos(theta)),
            np.float32(np.sin(theta)),
            np.float32(cx_s),
            np.float32(cy_s),
            np.float32(sc_s),
            np.int32(w_img),
            np.int32(h_img),
            np.int32(n_voxels),
        )

        # Progress (don't read back occupied every time — that would kill perf)
        pct = int((vi + 1) / n_views * 100)
        if progress_cb and ((vi + 1) % max(1, n_views // 50) == 0
                            or vi == 0 or vi + 1 == n_views):
            elapsed = time.time() - t0
            rate = (vi + 1) / max(elapsed, 0.001)
            eta = (n_views - vi - 1) / rate
            msg = (f"[GPU] View {vi+1}/{n_views}  "
                   f"({theta_deg:7.2f}°)  "
                   f"[{elapsed:.1f}s / ~{eta:.0f}s left]")
            progress_cb(pct, msg)

    # ---- Download result ----
    queue.finish()
    cl.enqueue_copy(queue, occupied_host, buf_occ, is_blocking=True)

    # Cleanup GPU
    for buf in [buf_X, buf_Y, buf_Z, buf_occ, buf_mask]:
        if buf is not None:
            buf.release()
    pool.shutdown(wait=False)

    elapsed = time.time() - t0
    voxel_grid = occupied_host.astype(bool).reshape(nx, ny, nz)
    return voxel_grid, bounds, (nx, ny, nz), elapsed


# ═══════════════════════════════════════════════════════════════════════════
#  Voxel carving — CPU path  (fully vectorised NumPy, threaded mask I/O)
# ═══════════════════════════════════════════════════════════════════════════

def _carve(
    file_list, angles_deg, params, cfg: EngineConfig, tilt_deg: float,
    progress_cb: Optional[Callable] = None,
):
    """Voxel carving with parallel mask loading and vectorised projection."""
    cx_s  = params["cx"]   # already at mask_scale via _analyze_silhouettes
    cy_s  = params["cy"]
    sc_s  = params["scale"]
    R     = params["max_radius"]
    ymin  = params["y_min"]
    ymax  = params["y_max"]

    pad = R * 0.15
    bounds = ((-R - pad, R + pad), (ymin, ymax), (-R - pad, R + pad))
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds

    nx = nz = cfg.resolution
    ny = max(int(cfg.resolution * (yhi - ylo) / (xhi - xlo)), 10)

    x = np.linspace(xlo, xhi, nx, dtype=np.float32)
    y = np.linspace(ylo, yhi, ny, dtype=np.float32)
    z = np.linspace(zlo, zhi, nz, dtype=np.float32)

    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    Xf = Xg.ravel(); Yf = Yg.ravel(); Zf = Zg.ravel()
    del Xg, Yg, Zg

    occupied = np.ones(Xf.size, dtype=bool)

    view_idx = list(range(0, len(file_list), cfg.skip_views))
    n_views = len(view_idx)

    # Pre-fetch masks in a sliding window using thread pool
    PREFETCH = min(cfg.n_workers * 2, n_views)

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, Future

    pool = ThreadPoolExecutor(max_workers=cfg.n_workers)
    prefetch_q: deque[Tuple[int, Future]] = deque()

    def submit_load(vi_pos):
        idx = view_idx[vi_pos]
        fp = file_list[idx][0]
        return pool.submit(_load_mask, fp, cfg.mask_scale, tilt_deg)

    # Seed the prefetch queue
    for i in range(min(PREFETCH, n_views)):
        prefetch_q.append((i, submit_load(i)))

    t0 = time.time()
    log_lines = []

    for vi in range(n_views):
        # Get next pre-loaded mask
        qi, fut = prefetch_q.popleft()
        assert qi == vi
        mask = fut.result()

        # Submit next prefetch
        next_i = vi + PREFETCH
        if next_i < n_views:
            prefetch_q.append((next_i, submit_load(next_i)))

        idx = view_idx[vi]
        theta_deg = angles_deg[idx]
        if cfg.flip_rotation:
            theta_deg = -theta_deg
        theta = np.radians(theta_deg).astype(np.float32)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        h_img, w_img = mask.shape

        occ_idx = np.where(occupied)[0]
        if len(occ_idx) == 0:
            break

        # Orthographic projection
        xp = Xf[occ_idx] * cos_t + Zf[occ_idx] * sin_t
        yp = Yf[occ_idx]

        u = (sc_s * xp + cx_s).astype(np.int32)
        v = (-sc_s * yp + cy_s).astype(np.int32)

        ib = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)

        in_sil = np.zeros(len(occ_idx), dtype=bool)
        ib_where = np.where(ib)[0]
        if len(ib_where) > 0:
            in_sil[ib_where] = mask[v[ib_where], u[ib_where]]

        carved = ib & ~in_sil
        occupied[occ_idx[carved]] = False

        # Progress reporting
        pct = int((vi + 1) / n_views * 100)
        if progress_cb and ((vi + 1) % max(1, n_views // 50) == 0
                            or vi == 0 or vi + 1 == n_views):
            elapsed = time.time() - t0
            rate = (vi + 1) / max(elapsed, 0.001)
            eta = (n_views - vi - 1) / rate
            msg = (f"View {vi+1}/{n_views}  "
                   f"({np.sum(occupied):,} voxels)  "
                   f"[{elapsed:.0f}s / ~{eta:.0f}s left]")
            progress_cb(pct, msg)

    pool.shutdown(wait=False)

    elapsed = time.time() - t0
    voxel_grid = occupied.reshape(nx, ny, nz)
    return voxel_grid, bounds, (nx, ny, nz), elapsed


# ═══════════════════════════════════════════════════════════════════════════
#  Export helpers
# ═══════════════════════════════════════════════════════════════════════════

def _export_mesh(voxel_grid, bounds, path, sigma=0.5):
    """Marching cubes → OBJ file.  Returns (n_verts, n_faces) or None."""
    if not HAS_SKIMAGE:
        return None

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    nx, ny, nz = voxel_grid.shape

    vol = np.pad(voxel_grid.astype(np.float32), 1, constant_values=0)
    if HAS_SCIPY and sigma > 0:
        vol = gaussian_filter(vol, sigma=sigma)

    try:
        verts, faces, normals, _ = sk_measure.marching_cubes(vol, level=0.5)
    except Exception:
        return None

    verts -= 1.0
    verts[:, 0] = verts[:, 0] / max(nx - 1, 1) * (xhi - xlo) + xlo
    verts[:, 1] = verts[:, 1] / max(ny - 1, 1) * (yhi - ylo) + ylo
    verts[:, 2] = verts[:, 2] / max(nz - 1, 1) * (zhi - zlo) + zlo

    base, _ = os.path.splitext(path)
    obj_path = base + ".obj"

    with open(obj_path, "w") as f:
        f.write(f"# Visual Hull — {len(verts)} verts, {len(faces)} faces\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1}//{face[0]+1} "
                    f"{face[1]+1}//{face[1]+1} "
                    f"{face[2]+1}//{face[2]+1}\n")

    return obj_path, len(verts), len(faces)


def _export_ply(voxel_grid, bounds, path):
    """Point cloud (ASCII PLY)."""
    nx, ny, nz = voxel_grid.shape
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)
    z = np.linspace(zlo, zhi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    occ = voxel_grid.ravel().astype(bool)
    pts = np.column_stack([X.ravel()[occ], Y.ravel()[occ], Z.ravel()[occ]])

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return path, len(pts)


def _save_npz(voxel_grid, bounds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, voxel_grid=voxel_grid,
                        volume_bounds=np.array(bounds))
    return path


def _save_preview(voxel_grid, bounds, out_dir, tool_id, max_pts=50000):
    """4-panel scatter preview → PNG."""
    if not HAS_MATPLOTLIB:
        return None
    os.makedirs(out_dir, exist_ok=True)
    nx, ny, nz = voxel_grid.shape
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)
    z = np.linspace(zlo, zhi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    occ = voxel_grid.ravel().astype(bool)
    px, py, pz = X.ravel()[occ], Y.ravel()[occ], Z.ravel()[occ]
    n = len(px)
    if n == 0:
        return None
    if n > max_pts:
        idx = np.random.default_rng(42).choice(n, max_pts, replace=False)
        px, py, pz = px[idx], py[idx], pz[idx]

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(px, pz, py, s=0.3, alpha=0.2, c=py, cmap="viridis")
    ax1.set_xlabel("X"); ax1.set_ylabel("Z"); ax1.set_zlabel("Y")
    ax1.set_title("3D view")

    for i, (a, b, xl, yl, title) in enumerate([
        (px, pz, "X", "Z", "Top-down (XZ)"),
        (px, py, "X", "Y", "Front (XY)"),
        (pz, py, "Z", "Y", "Side (ZY)"),
    ], start=2):
        ax = fig.add_subplot(2, 2, i)
        ax.scatter(a, b, s=0.1, alpha=0.1, c="steelblue")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Visual Hull — {tool_id}  ({n:,} voxels)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tool_id}_visual_hull_preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_cross_sections(voxel_grid, bounds, out_dir, tool_id, n_slices=5):
    if not HAS_MATPLOTLIB:
        return None
    os.makedirs(out_dir, exist_ok=True)
    nx, ny, nz = voxel_grid.shape
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    si = np.linspace(0, ny - 1, n_slices + 2, dtype=int)[1:-1]
    fig, axes = plt.subplots(1, len(si), figsize=(4 * len(si), 4))
    if len(si) == 1:
        axes = [axes]
    for ax, yi in zip(axes, si):
        yv = ylo + yi / max(ny - 1, 1) * (yhi - ylo)
        ax.imshow(voxel_grid[:, yi, :].T, origin="lower", cmap="gray",
                  extent=[xlo, xhi, zlo, zhi], aspect="equal")
        ax.set_title(f"Y = {yv:.2f}"); ax.set_xlabel("X"); ax.set_ylabel("Z")
    fig.suptitle(f"Cross-sections — {tool_id}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tool_id}_cross_sections.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_visual_hull(
    cfg: EngineConfig,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Execute a full Visual Hull reconstruction.

    Args:
        cfg         : EngineConfig with all parameters (call .resolve() first)
        progress_cb : fn(percent: int, message: str) — progress updates
        log_cb      : fn(text: str) — log lines for the console widget

    Returns:
        dict with keys: obj_path, ply_path, npz_path, preview_path,
                        cross_path, run_config_path, elapsed, grid_shape,
                        n_occupied, error (None on success)
    """
    def _log(msg):
        if log_cb:
            log_cb(msg)
    def _prog(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    cfg.resolve()
    os.makedirs(cfg.output_dir, exist_ok=True)
    prefix = f"{cfg.tool_id}_visual_hull"

    result = {
        "obj_path": None, "ply_path": None, "npz_path": None,
        "preview_path": None, "cross_path": None, "run_config_path": None,
        "elapsed": 0, "grid_shape": None, "n_occupied": 0, "error": None,
    }

    try:
        # 1. Discover masks
        _prog(0, "Discovering masks …")
        _log(f"Mask directory: {cfg.mask_dir}")
        file_list = _discover_mask_files(cfg.mask_dir)
        _log(f"Found {len(file_list)} mask files")

        # 2. Angles
        _prog(2, "Determining rotation angles …")
        if cfg.angle_mode == "filename":
            angles = _parse_angles_from_filenames(file_list)
            _log(f"Angles from filenames: {angles[0]:.2f}° → {angles[-1]:.2f}°  "
                 f"(step ≈ {np.median(np.diff(angles)):.3f}°)")
        elif cfg.angle_mode == "uniform_360":
            angles = _uniform_angles(len(file_list), 360.0)
            _log(f"Uniform 360°: step = {360/len(file_list):.4f}°")
        elif cfg.angle_mode == "uniform_363":
            angles = _uniform_angles(len(file_list), 363.0)
            _log(f"Uniform 363°: step = {363/len(file_list):.4f}°")
        else:
            angles = _parse_angles_from_filenames(file_list)

        # 2b. Symmetry — use only the first half of images (≈180°)
        if cfg.symmetry_half:
            n_total = len(file_list)
            n_half = n_total // 2
            file_list = file_list[:n_half]
            angles = angles[:n_half]
            _log(f"Symmetry mode: using first {n_half} of {n_total} images "
                 f"(≈{angles[-1]:.2f}° span)")

        # 3. Tilt
        _prog(3, "Resolving tilt correction …")
        tilt_deg = resolve_tilt_angle(cfg.tilt_correction,
                                       cfg.tool_id, cfg.tilt_meta_dir)
        _log(f"Tilt correction: {tilt_deg:.4f}°"
             f"{'  (OFF)' if abs(tilt_deg) < 1e-6 else '  (ACTIVE)'}")

        # 4. Silhouette analysis
        _prog(5, "Analysing silhouettes …")
        params = _analyze_silhouettes(file_list, tilt_deg, cfg.mask_scale,
                                       cfg.n_workers)
        _log(f"Image {params['img_w']}×{params['img_h']}  "
             f"cx={params['cx']:.0f}  cy={params['cy']:.0f}  "
             f"scale={params['scale']:.0f} px/unit")

        # 5. Carving (GPU → CPU auto-fallback)
        use_gpu = cfg.use_gpu
        gpu_ctx, gpu_info = _get_opencl_gpu_context() if use_gpu else (None, "disabled")
        if gpu_ctx is not None:
            _prog(8, f"Starting GPU voxel carving on {gpu_info} …")
            _log(f"GPU device: {gpu_info}")
            try:
                voxel_grid, bounds, shape, carve_elapsed = _carve_gpu(
                    file_list, angles, params, cfg, tilt_deg,
                    progress_cb=lambda p, m: _prog(8 + int(p * 0.80), m),
                )
            except Exception as gpu_err:
                _log(f"GPU carving failed ({gpu_err}), falling back to CPU …")
                voxel_grid, bounds, shape, carve_elapsed = _carve(
                    file_list, angles, params, cfg, tilt_deg,
                    progress_cb=lambda p, m: _prog(8 + int(p * 0.80), m),
                )
        else:
            reason = gpu_info if not use_gpu else gpu_info
            _prog(8, f"Starting CPU voxel carving ({reason}) …")
            _log(f"GPU not used: {reason}")
            voxel_grid, bounds, shape, carve_elapsed = _carve(
                file_list, angles, params, cfg, tilt_deg,
                progress_cb=lambda p, m: _prog(8 + int(p * 0.80), m),
            )
        n_occ = int(np.sum(voxel_grid))
        _log(f"Carving done in {carve_elapsed:.1f}s — "
             f"{n_occ:,} / {int(np.prod(shape)):,} voxels occupied")
        result["grid_shape"] = shape
        result["n_occupied"] = n_occ

        # 6. Export voxels (always)
        _prog(90, "Saving voxel grid …")
        npz_path = os.path.join(cfg.output_dir, f"{prefix}_voxels.npz")
        _save_npz(voxel_grid, bounds, npz_path)
        result["npz_path"] = npz_path
        _log(f"Voxels → {npz_path}")

        # 7. Mesh
        if cfg.export_mesh:
            _prog(92, "Extracting mesh (marching cubes) …")
            mesh_path = os.path.join(cfg.output_dir, f"{prefix}.obj")
            mesh_res = _export_mesh(voxel_grid, bounds, mesh_path,
                                    sigma=cfg.smooth_sigma)
            if mesh_res:
                result["obj_path"] = mesh_res[0]
                _log(f"Mesh → {mesh_res[0]}  "
                     f"({mesh_res[1]:,} verts, {mesh_res[2]:,} faces)")
            else:
                _log("Mesh export skipped (scikit-image not available)")

        # 8. Point cloud
        if cfg.export_pointcloud:
            _prog(95, "Exporting point cloud …")
            ply_path = os.path.join(cfg.output_dir, f"{prefix}.ply")
            ply_res = _export_ply(voxel_grid, bounds, ply_path)
            result["ply_path"] = ply_res[0]
            _log(f"PLY → {ply_res[0]}  ({ply_res[1]:,} points)")

        # 9. Visualisation
        if cfg.export_viz:
            _prog(97, "Creating preview images …")
            result["preview_path"] = _save_preview(voxel_grid, bounds,
                                                    cfg.output_dir, cfg.tool_id)
            result["cross_path"] = _save_cross_sections(voxel_grid, bounds,
                                                         cfg.output_dir, cfg.tool_id)
            if result["preview_path"]:
                _log(f"Preview → {result['preview_path']}")

        # 10. Run config JSON
        run_config = {
            **asdict(cfg),
            "tilt_angle_applied_deg": tilt_deg,
            "grid_shape": list(shape),
            "volume_bounds": [list(b) for b in bounds],
            "occupied_voxels": n_occ,
            "total_voxels": int(np.prod(shape)),
            "carve_elapsed_s": round(carve_elapsed, 2),
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        cfg_path = os.path.join(cfg.output_dir, "run_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        result["run_config_path"] = cfg_path

        result["elapsed"] = carve_elapsed
        _prog(100, "Done!")
        _log("=" * 50)
        _log(f"Done!  Output → {cfg.output_dir}")

    except Exception as e:
        result["error"] = str(e)
        _log(f"ERROR: {e}")
        _prog(100, f"Error: {e}")
        import traceback
        _log(traceback.format_exc())

    return result
