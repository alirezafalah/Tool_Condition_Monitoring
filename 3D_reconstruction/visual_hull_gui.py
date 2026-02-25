#!/usr/bin/env python3
"""
Visual Hull GUI — Professional tkinter interface for Shape from Silhouette.
===========================================================================

Launch this file to open the GUI.  No command-line arguments required.

    python visual_hull_gui.py

Requirements (all in standard Anaconda / pip):
    numpy, Pillow, scikit-image, scipy, matplotlib, tkinter (built-in)
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Ensure the engine module is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from visual_hull_engine import (
    EngineConfig, discover_tools, run_visual_hull, get_gpu_info,
    MASKS_DIR, OUTPUT_ROOT, MESHLAB_EXE, MAX_WORKERS,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Theme & style constants
# ═══════════════════════════════════════════════════════════════════════════

BG          = "#1e1e2e"      # dark background
BG_CARD     = "#2a2a3c"      # card / frame bg
BG_INPUT    = "#363650"      # entry / combobox bg
FG          = "#cdd6f4"      # normal text
FG_DIM      = "#7f849c"      # secondary text
FG_ACCENT   = "#89b4fa"      # blue accent
FG_GREEN    = "#a6e3a1"      # success green
FG_YELLOW   = "#f9e2af"      # warning yellow
FG_RED      = "#f38ba8"      # error red
BORDER      = "#45475a"
BTN_BG      = "#89b4fa"
BTN_FG      = "#1e1e2e"
BTN_HOVER   = "#b4d0fb"
BTN_RUN_BG  = "#a6e3a1"
BTN_RUN_FG  = "#1e1e2e"
FONT_FAMILY = "Segoe UI"
FONT        = (FONT_FAMILY, 10)
FONT_BOLD   = (FONT_FAMILY, 10, "bold")
FONT_SMALL  = (FONT_FAMILY, 9)
FONT_TITLE  = (FONT_FAMILY, 14, "bold")
FONT_MONO   = ("Cascadia Code", 9)


class ToolTip:
    """Hover tooltip for any widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg="#45475a")
        label = tk.Label(tw, text=self.text, justify="left",
                         bg="#45475a", fg=FG, font=FONT_SMALL,
                         padx=8, pady=4, wraplength=350)
        label.pack()

    def hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ═══════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════

class VisualHullApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visual Hull — Shape from Silhouette")
        self.configure(bg=BG)
        self.minsize(820, 720)
        self.geometry("880x820")
        self._center_window()

        # State
        self._running = False
        self._result = None

        # Variables (tk vars)
        self._tools = discover_tools()

        self.var_tool      = tk.StringVar(value=self._tools[0] if self._tools else "tool002")
        self.var_mask_dir  = tk.StringVar()
        self.var_output_dir= tk.StringVar()
        self.var_angle     = tk.StringVar(value="filename")
        self.var_resolution= tk.IntVar(value=200)
        self.var_skip      = tk.IntVar(value=1)
        self.var_scale     = tk.DoubleVar(value=0.5)
        self.var_flip      = tk.BooleanVar(value=False)
        self.var_symmetry  = tk.BooleanVar(value=False)
        self.var_sigma     = tk.DoubleVar(value=0.5)
        self.var_tilt_mode = tk.StringVar(value="off")
        self.var_tilt_angle= tk.DoubleVar(value=0.0)
        self.var_mesh      = tk.BooleanVar(value=True)
        self.var_ply       = tk.BooleanVar(value=True)
        self.var_viz       = tk.BooleanVar(value=True)
        self.var_workers   = tk.IntVar(value=MAX_WORKERS)
        self.var_gpu       = tk.BooleanVar(value=True)

        # Probe GPU once at startup
        self._gpu_info = get_gpu_info()

        self._build_ui()
        self._on_tool_changed()  # populate paths

    # -------------------------------------------------------------------
    #  Window helpers
    # -------------------------------------------------------------------

    def _center_window(self):
        self.update_idletasks()
        w, h = 880, 820
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # -------------------------------------------------------------------
    #  UI construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        # Configure ttk styles
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Card.TFrame",       background=BG_CARD)
        style.configure("Dark.TFrame",       background=BG)
        style.configure("Card.TLabelframe",  background=BG_CARD, foreground=FG_ACCENT,
                         font=FONT_BOLD)
        style.configure("Card.TLabelframe.Label", background=BG_CARD,
                         foreground=FG_ACCENT, font=FONT_BOLD)

        style.configure("TLabel",  background=BG_CARD, foreground=FG, font=FONT)
        style.configure("Dim.TLabel", background=BG_CARD, foreground=FG_DIM,
                         font=FONT_SMALL)
        style.configure("Title.TLabel", background=BG, foreground=FG_ACCENT,
                         font=FONT_TITLE)

        style.configure("TCheckbutton", background=BG_CARD, foreground=FG,
                         font=FONT)
        style.map("TCheckbutton",
                  background=[("active", BG_CARD)],
                  foreground=[("active", FG)])

        style.configure("TCombobox", fieldbackground=BG_INPUT,
                         background=BG_INPUT, foreground=FG, font=FONT,
                         arrowcolor=FG_ACCENT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", BG_INPUT)],
                  selectbackground=[("readonly", BG_INPUT)],
                  selectforeground=[("readonly", FG)])

        style.configure("TSpinbox", fieldbackground=BG_INPUT,
                         background=BG_INPUT, foreground=FG, font=FONT,
                         arrowcolor=FG_ACCENT)

        style.configure("Accent.TButton", background=BTN_BG, foreground=BTN_FG,
                         font=FONT_BOLD, padding=(14, 6))
        style.map("Accent.TButton",
                  background=[("active", BTN_HOVER), ("disabled", BORDER)])

        style.configure("Run.TButton", background=BTN_RUN_BG,
                         foreground=BTN_RUN_FG, font=(FONT_FAMILY, 12, "bold"),
                         padding=(20, 10))
        style.map("Run.TButton",
                  background=[("active", "#76d68a"), ("disabled", BORDER)])

        style.configure("Small.TButton", background=BG_INPUT, foreground=FG,
                         font=FONT_SMALL, padding=(8, 2))
        style.map("Small.TButton",
                  background=[("active", BORDER)])

        # Progress bar
        style.configure("green.Horizontal.TProgressbar",
                         troughcolor=BG_INPUT, background=FG_GREEN,
                         thickness=18)

        # Main scroll canvas
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        # Title bar
        title_frame = tk.Frame(outer, bg=BG, pady=8)
        title_frame.pack(fill="x")
        ttk.Label(title_frame, text="  Visual Hull  — Shape from Silhouette",
                  style="Title.TLabel").pack(side="left", padx=10)
        ttk.Label(title_frame, text="3D Reconstruction",
                  style="Dim.TLabel").pack(side="left", padx=(0, 10))

        # Scrollable content
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self._content = content = tk.Frame(canvas, bg=BG)
        content.bind("<Configure>",
                     lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Cards ---
        self._build_tool_card(content)
        self._build_angles_card(content)
        self._build_carving_card(content)
        self._build_tilt_card(content)
        self._build_gpu_card(content)
        self._build_export_card(content)
        self._build_run_card(content)
        self._build_log_card(content)
        self._build_results_card(content)

    # --- Tool selection card ---
    def _build_tool_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Tool Selection  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=(8, 4))

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")

        ttk.Label(row, text="Tool:").pack(side="left", padx=(0, 8))
        cb = ttk.Combobox(row, textvariable=self.var_tool,
                          values=self._tools, state="readonly", width=14)
        cb.pack(side="left", padx=(0, 16))
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_tool_changed())
        ToolTip(cb, "Select the tool whose masks you want to reconstruct.")

        ttk.Label(row, text="Mask dir:").pack(side="left", padx=(0, 4))
        mask_entry = ttk.Entry(row, textvariable=self.var_mask_dir, width=44,
                               font=FONT_SMALL)
        mask_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="…", style="Small.TButton", width=3,
                   command=self._browse_mask_dir).pack(side="left")
        ToolTip(mask_entry, "Path to the folder containing binary mask images (.tiff).\nAuto-filled when you pick a tool.")

        row2 = ttk.Frame(frame, style="Card.TFrame")
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Output dir:").pack(side="left", padx=(0, 4))
        out_entry = ttk.Entry(row2, textvariable=self.var_output_dir, width=54,
                              font=FONT_SMALL)
        out_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row2, text="…", style="Small.TButton", width=3,
                   command=self._browse_output_dir).pack(side="left")
        ttk.Button(row2, text="Open", style="Small.TButton", width=5,
                   command=self._open_output_dir).pack(side="left", padx=(4, 0))
        ToolTip(out_entry, "Where results will be saved.  Auto-set to output/<tool_id>/.")

    # --- Angle mode card ---
    def _build_angles_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Rotation Angles  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=4)

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")

        ttk.Label(row, text="Angle mode:").pack(side="left", padx=(0, 8))
        modes = [
            ("From filenames", "filename"),
            ("Uniform 360°",   "uniform_360"),
            ("Uniform 363°",   "uniform_363"),
        ]
        for text, val in modes:
            rb = ttk.Radiobutton(row, text=text, variable=self.var_angle,
                                 value=val)
            rb.pack(side="left", padx=(0, 12))

        ToolTip(row,
            "filename — parse degree values from file names (e.g. 0045.50_degrees.tiff)\n"
            "Uniform 360° — treat N images as uniformly spaced over 360°\n"
            "Uniform 363° — treat N images as uniformly spaced over 363°  "
            "(use this if your 363 frames cover 363 degrees)")

        ttk.Checkbutton(row, text="Flip rotation direction",
                        variable=self.var_flip).pack(side="right")

        row2 = ttk.Frame(frame, style="Card.TFrame")
        row2.pack(fill="x", pady=(6, 0))
        sym_cb = ttk.Checkbutton(row2, text="Symmetry — use only first half of images (180°)",
                                 variable=self.var_symmetry)
        sym_cb.pack(side="left")
        ToolTip(sym_cb,
            "Enable for symmetric tools (e.g. 2-edge drills).\n"
            "Uses only the first half of images (≈180°) since the\n"
            "other side is a mirror of the first.\n\n"
            "Example: 368 images × 360° → 184 images × ≈180°.")

    # --- Carving parameters card ---
    def _build_carving_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Carving Parameters  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=4)

        grid = ttk.Frame(frame, style="Card.TFrame")
        grid.pack(fill="x")

        # Resolution
        ttk.Label(grid, text="Resolution (voxels):").grid(row=0, column=0,
                   sticky="w", padx=(0, 8), pady=2)
        res_spin = ttk.Spinbox(grid, textvariable=self.var_resolution,
                               from_=50, to=600, increment=50, width=8)
        res_spin.grid(row=0, column=1, sticky="w", pady=2)
        ToolTip(res_spin,
            "Voxels per X/Z axis.  Higher = finer detail but slower.\n"
            "100 = fast preview  |  200 = good  |  400 = high detail")

        # Skip views
        ttk.Label(grid, text="Skip views (every N-th):").grid(row=0, column=2,
                   sticky="w", padx=(24, 8), pady=2)
        skip_spin = ttk.Spinbox(grid, textvariable=self.var_skip,
                                from_=1, to=30, increment=1, width=6)
        skip_spin.grid(row=0, column=3, sticky="w", pady=2)
        ToolTip(skip_spin,
            "Use every N-th view.  1 = all views (best quality).\n"
            "3–5 = fast preview with reasonable quality.")

        # Mask scale
        ttk.Label(grid, text="Mask scale:").grid(row=1, column=0,
                   sticky="w", padx=(0, 8), pady=2)
        sc_spin = ttk.Spinbox(grid, textvariable=self.var_scale,
                              from_=0.1, to=1.0, increment=0.1, width=8,
                              format="%.1f")
        sc_spin.grid(row=1, column=1, sticky="w", pady=2)
        ToolTip(sc_spin,
            "Downscale factor for masks.\n"
            "0.5 = half resolution (faster, uses less RAM).\n"
            "1.0 = full resolution.")

        # Workers
        ttk.Label(grid, text="CPU threads:").grid(row=1, column=2,
                   sticky="w", padx=(24, 8), pady=2)
        w_spin = ttk.Spinbox(grid, textvariable=self.var_workers,
                             from_=1, to=16, increment=1, width=6)
        w_spin.grid(row=1, column=3, sticky="w", pady=2)
        ToolTip(w_spin,
            "Threads for parallel mask loading.\n"
            "12 is optimal for your 16-thread CPU (leaves headroom for OS + GUI).")

        # Smooth sigma
        ttk.Label(grid, text="Mesh smooth σ:").grid(row=2, column=0,
                   sticky="w", padx=(0, 8), pady=2)
        sig_spin = ttk.Spinbox(grid, textvariable=self.var_sigma,
                               from_=0.0, to=3.0, increment=0.1, width=8,
                               format="%.1f")
        sig_spin.grid(row=2, column=1, sticky="w", pady=2)
        ToolTip(sig_spin,
            "Gaussian smoothing applied to voxel volume before mesh extraction.\n"
            "0 = no smoothing (blocky)  |  0.5 = light  |  1.5 = very smooth")

    # --- Tilt correction card ---
    def _build_tilt_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Tilt / Axis Correction  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=4)

        row1 = ttk.Frame(frame, style="Card.TFrame")
        row1.pack(fill="x")

        ttk.Label(row1, text="Mode:").pack(side="left", padx=(0, 8))
        for text, val in [("Off", "off"), ("Auto (master mask)", "auto"),
                          ("Manual", "manual")]:
            rb = ttk.Radiobutton(row1, text=text, variable=self.var_tilt_mode,
                                 value=val, command=self._on_tilt_mode_changed)
            rb.pack(side="left", padx=(0, 12))

        ToolTip(row1,
            "Off — assume rotation axis is perfectly vertical in the image.\n"
            "Auto — load the tilt angle from master-mask metadata JSON.\n"
            "Manual — enter a custom tilt angle below.")

        row2 = ttk.Frame(frame, style="Card.TFrame")
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Tilt angle (°):").pack(side="left", padx=(0, 8))
        self._tilt_spin = ttk.Spinbox(
            row2, textvariable=self.var_tilt_angle,
            from_=-10.0, to=10.0, increment=0.1, width=10, format="%.4f")
        self._tilt_spin.pack(side="left")
        self._tilt_spin.configure(state="disabled")
        ToolTip(self._tilt_spin,
            "Counter-rotation angle in degrees to correct axis tilt.\n"
            "Only active in Manual mode.")

        ttk.Label(row2, text="(positive = CCW correction)",
                  style="Dim.TLabel").pack(side="left", padx=(8, 0))

    # --- GPU card ---
    def _build_gpu_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  GPU Acceleration  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=4)

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")

        gpu = self._gpu_info
        if gpu["available"]:
            self.var_gpu.set(True)
            ttk.Checkbutton(row, text="Use GPU (OpenCL)",
                            variable=self.var_gpu).pack(side="left", padx=(0, 16))
            info_text = (f"{gpu['device']}  —  "
                         f"{gpu['mem_mb']} MB  |  "
                         f"{gpu['compute_units']} compute units  |  "
                         f"max workgroup {gpu['max_work_group']}")
            lbl = ttk.Label(row, text=info_text, style="Dim.TLabel")
            lbl.pack(side="left", fill="x")
            ToolTip(row,
                "Run the voxel carving on your Intel Iris Xe GPU via OpenCL.\n"
                "Voxel data stays on the GPU — only the mask image is uploaded per view.\n"
                "Typically 5–20× faster than CPU for large grids.\n"
                "Uncheck to force CPU-only mode (numpy vectorized).")
        else:
            self.var_gpu.set(False)
            ttk.Checkbutton(row, text="Use GPU (OpenCL)",
                            variable=self.var_gpu,
                            state="disabled").pack(side="left", padx=(0, 16))
            ttk.Label(row, text=f"Not available: {gpu['device']}",
                      foreground=FG_YELLOW,
                      style="Dim.TLabel").pack(side="left")

    # --- Export options card ---
    def _build_export_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Export Options  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=4)

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")

        ttk.Checkbutton(row, text="Mesh (.obj)", variable=self.var_mesh
                        ).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row, text="Point cloud (.ply)", variable=self.var_ply
                        ).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row, text="Preview images (.png)",
                        variable=self.var_viz).pack(side="left")

        ttk.Label(row, text="(Voxel grid .npz is always saved)",
                  style="Dim.TLabel").pack(side="right")

    # --- Run card ---
    def _build_run_card(self, parent):
        frame = ttk.Frame(parent, style="Dark.TFrame")
        frame.pack(fill="x", padx=8, pady=(8, 4))

        self.btn_run = ttk.Button(frame, text="▶  Run Reconstruction",
                                  style="Run.TButton",
                                  command=self._on_run)
        self.btn_run.pack(side="left", padx=(0, 16))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100,
            style="green.Horizontal.TProgressbar", length=300)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.lbl_status = ttk.Label(frame, text="Ready", style="Dim.TLabel")
        self.lbl_status.configure(background=BG)
        self.lbl_status.pack(side="left")

    # --- Log card ---
    def _build_log_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Console Log  ",
                                style="Card.TLabelframe", padding=8)
        frame.pack(fill="x", padx=8, pady=4)

        self.log_text = tk.Text(frame, height=12, bg="#181825", fg=FG,
                                insertbackground=FG, font=FONT_MONO,
                                relief="flat", borderwidth=0, wrap="word",
                                state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # Scroll
        sb = ttk.Scrollbar(frame, orient="vertical",
                           command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

    # --- Results / viewer card ---
    def _build_results_card(self, parent):
        frame = ttk.LabelFrame(parent, text="  Results & Viewer  ",
                                style="Card.TLabelframe", padding=12)
        frame.pack(fill="x", padx=8, pady=(4, 12))

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")

        self.btn_meshlab = ttk.Button(row, text="Open in MeshLab",
                                      style="Accent.TButton",
                                      command=self._open_meshlab,
                                      state="disabled")
        self.btn_meshlab.pack(side="left", padx=(0, 8))
        ToolTip(self.btn_meshlab,
            "Open the generated .obj mesh in MeshLab for interactive 3D viewing.")

        self.btn_preview = ttk.Button(row, text="View Preview Image",
                                      style="Accent.TButton",
                                      command=self._open_preview,
                                      state="disabled")
        self.btn_preview.pack(side="left", padx=(0, 8))

        self.btn_open_folder = ttk.Button(row, text="Open Output Folder",
                                          style="Small.TButton",
                                          command=self._open_output_dir)
        self.btn_open_folder.pack(side="left", padx=(0, 8))

        self.lbl_result_info = ttk.Label(row, text="", style="Dim.TLabel")
        self.lbl_result_info.pack(side="left", fill="x", expand=True)

    # -------------------------------------------------------------------
    #  Callbacks
    # -------------------------------------------------------------------

    def _on_tool_changed(self):
        tid = self.var_tool.get()
        self.var_mask_dir.set(os.path.join(MASKS_DIR, f"{tid}_final_masks"))
        self.var_output_dir.set(os.path.join(OUTPUT_ROOT, tid))

    def _on_tilt_mode_changed(self):
        mode = self.var_tilt_mode.get()
        if mode == "manual":
            self._tilt_spin.configure(state="normal")
        else:
            self._tilt_spin.configure(state="disabled")

    def _browse_mask_dir(self):
        d = filedialog.askdirectory(initialdir=MASKS_DIR,
                                    title="Select Mask Folder")
        if d:
            self.var_mask_dir.set(d)

    def _browse_output_dir(self):
        d = filedialog.askdirectory(initialdir=OUTPUT_ROOT,
                                    title="Select Output Folder")
        if d:
            self.var_output_dir.set(d)

    def _open_output_dir(self):
        d = self.var_output_dir.get()
        if os.path.isdir(d):
            os.startfile(d)
        else:
            messagebox.showinfo("Output", f"Folder does not exist yet:\n{d}")

    # -------------------------------------------------------------------
    #  Build config from GUI state
    # -------------------------------------------------------------------

    def _build_config(self) -> EngineConfig:
        tilt_mode = self.var_tilt_mode.get()
        if tilt_mode == "off":
            tilt_val = "off"
        elif tilt_mode == "auto":
            tilt_val = "auto"
        else:
            tilt_val = str(self.var_tilt_angle.get())

        cfg = EngineConfig(
            tool_id         = self.var_tool.get(),
            mask_dir        = self.var_mask_dir.get(),
            output_dir      = self.var_output_dir.get(),
            angle_mode      = self.var_angle.get(),
            resolution      = self.var_resolution.get(),
            skip_views      = max(1, self.var_skip.get()),
            mask_scale      = max(0.1, min(1.0, self.var_scale.get())),
            flip_rotation   = self.var_flip.get(),
            smooth_sigma    = max(0.0, self.var_sigma.get()),
            tilt_correction = tilt_val,
            symmetry_half   = self.var_symmetry.get(),
            export_mesh     = self.var_mesh.get(),
            export_pointcloud = self.var_ply.get(),
            export_viz      = self.var_viz.get(),
            use_gpu         = self.var_gpu.get(),
            n_workers       = max(1, self.var_workers.get()),
        )
        return cfg

    # -------------------------------------------------------------------
    #  Logging
    # -------------------------------------------------------------------

    def _log(self, text):
        """Append text to the log widget (thread-safe via after())."""
        def _do():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, _do)

    def _set_progress(self, pct, msg=""):
        def _do():
            self.progress_var.set(pct)
            if msg:
                self.lbl_status.configure(text=msg)
        self.after(0, _do)

    # -------------------------------------------------------------------
    #  Run reconstruction
    # -------------------------------------------------------------------

    def _on_run(self):
        if self._running:
            return

        # Validate
        mask_dir = self.var_mask_dir.get()
        if not os.path.isdir(mask_dir):
            messagebox.showerror("Error",
                f"Mask directory not found:\n{mask_dir}")
            return

        cfg = self._build_config()
        self._running = True
        self.btn_run.configure(state="disabled")
        self.btn_meshlab.configure(state="disabled")
        self.btn_preview.configure(state="disabled")
        self.lbl_result_info.configure(text="")
        self.progress_var.set(0)

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self._log(f"Starting reconstruction for {cfg.tool_id} ...")
        self._log(f"  Resolution: {cfg.resolution}  Skip: {cfg.skip_views}  "
                  f"Scale: {cfg.mask_scale}  Workers: {cfg.n_workers}")
        self._log(f"  Angle mode: {cfg.angle_mode}  "
                  f"Tilt: {cfg.tilt_correction}  Flip: {cfg.flip_rotation}  "
                  f"Symmetry: {cfg.symmetry_half}")
        self._log("")

        def _worker():
            result = run_visual_hull(
                cfg,
                progress_cb=self._set_progress,
                log_cb=self._log,
            )
            self.after(0, lambda: self._on_run_done(result))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_run_done(self, result):
        self._running = False
        self._result = result
        self.btn_run.configure(state="normal")

        if result.get("error"):
            self.lbl_status.configure(text="Error!")
            self._log(f"\nERROR: {result['error']}")
            messagebox.showerror("Reconstruction Error", result["error"])
            return

        self.lbl_status.configure(text="Done!")

        # Enable viewer buttons
        if result.get("obj_path") and os.path.isfile(result["obj_path"]):
            self.btn_meshlab.configure(state="normal")

        if result.get("preview_path") and os.path.isfile(result["preview_path"]):
            self.btn_preview.configure(state="normal")

        gs = result.get("grid_shape", (0, 0, 0))
        info = (f"{result.get('n_occupied', 0):,} voxels  "
                f"({gs[0]}×{gs[1]}×{gs[2]})  "
                f"in {result.get('elapsed', 0):.1f}s")
        self.lbl_result_info.configure(text=info)

    # -------------------------------------------------------------------
    #  Viewers
    # -------------------------------------------------------------------

    def _open_meshlab(self):
        if not self._result or not self._result.get("obj_path"):
            return
        obj = self._result["obj_path"]
        if not os.path.isfile(obj):
            messagebox.showinfo("MeshLab", f"OBJ file not found:\n{obj}")
            return

        if os.path.isfile(MESHLAB_EXE):
            subprocess.Popen([MESHLAB_EXE, obj])
        else:
            # Try system association
            try:
                os.startfile(obj)
            except Exception:
                messagebox.showinfo(
                    "MeshLab",
                    f"MeshLab not found at:\n{MESHLAB_EXE}\n\n"
                    "Please install MeshLab or open the file manually:\n"
                    f"{obj}")

    def _open_preview(self):
        if not self._result or not self._result.get("preview_path"):
            return
        path = self._result["preview_path"]
        if os.path.isfile(path):
            os.startfile(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = VisualHullApp()
    app.mainloop()
