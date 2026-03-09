import os
import random
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "build_master_masks_all_two_edge_tools.py")

DEFAULT_BASE_DATA_DIR = (
    r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
)
DEFAULT_MASKS_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, "masks")
DEFAULT_MASKS_TILTED_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, "masks_tilted")


def recommended_worker_count():
    cpu_total = os.cpu_count() or 4
    reserve_cores = 2
    bounded = max(1, cpu_total - reserve_cores)
    fractional = max(1, int(round(cpu_total * 0.6)))
    return max(1, min(8, bounded, fractional))


# ============================================================================
# MATRIX COLOR SCHEME
# ============================================================================
BG_MAIN = "#020802"
BG_PANEL = "#031103"
BG_ENTRY = "#021102"
FG_MAIN = "#65ff7a"
FG_DIM = "#2abf49"
FG_WARN = "#b9ff3f"
BORDER = "#1c8c33"
BG_BTN = "#063012"
BG_BTN_ACTIVE = "#0b4721"
BG_LOG = "#010601"

FONT_TITLE = ("Consolas", 22, "bold")
FONT_LABEL = ("Consolas", 10, "bold")
FONT_ENTRY = ("Consolas", 10)
FONT_BTN = ("Consolas", 10, "bold")
FONT_LOG = ("Consolas", 10)
FONT_TAB = ("Consolas", 11, "bold")


# ============================================================================
# SHARED WIDGET HELPERS
# ============================================================================
def make_label(parent, text, fg=None):
    return tk.Label(parent, text=text, bg=parent["bg"], fg=fg or FG_DIM, font=FONT_LABEL)


def make_entry(parent, text_var, width=62):
    return tk.Entry(
        parent, textvariable=text_var, width=width,
        bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN,
        relief=tk.FLAT, font=FONT_ENTRY,
        highlightbackground=BORDER, highlightcolor=FG_WARN, highlightthickness=1,
    )


def make_button(parent, text, command, width=None, fg=None):
    return tk.Button(
        parent, text=text, command=command, width=width,
        bg=BG_BTN, fg=fg or FG_MAIN, activebackground=BG_BTN_ACTIVE,
        activeforeground="#dcffdc", relief=tk.FLAT, bd=0,
        font=FONT_BTN, padx=10, pady=6,
        highlightbackground=BORDER, highlightthickness=1,
    )


def make_log(parent, height=26):
    return ScrolledText(
        parent, wrap=tk.WORD, height=height,
        bg=BG_LOG, fg=FG_MAIN, insertbackground=FG_MAIN,
        relief=tk.FLAT, font=FONT_LOG,
        highlightbackground=BORDER, highlightcolor=FG_WARN, highlightthickness=1,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class MatrixPerspectiveGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MATRIX // Perspective & ROI Control")
        self.geometry("1220x800")
        self.minsize(1060, 720)
        self.configure(bg=BG_MAIN)

        self.process = None

        self.base_data_var = tk.StringVar(value=DEFAULT_BASE_DATA_DIR)
        self.masks_root_var = tk.StringVar(value=DEFAULT_MASKS_DIR)
        self.max_workers_var = tk.IntVar(value=recommended_worker_count())
        self.selected_count_var = tk.StringVar(value="Selected: 0")

        self.folder_names = []

        self._build_ui()
        self._load_mask_subfolders()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        title = tk.Label(
            self, text="MATRIX // PERSPECTIVE & ROI CONTROL",
            bg=BG_MAIN, fg=FG_MAIN, font=FONT_TITLE, pady=10,
        )
        title.pack(fill=tk.X)

        self.status_label = tk.Label(
            self, text="[ READY ]", bg=BG_MAIN, fg=FG_WARN,
            font=("Consolas", 11, "bold"),
        )
        self.status_label.pack(fill=tk.X)
        self._animate_status()

        # ---- Custom tab bar ----
        tab_bar = tk.Frame(self, bg=BG_MAIN)
        tab_bar.pack(fill=tk.X, padx=12, pady=(6, 0))

        self.tab_buttons = {}
        self.tab_frames = {}

        self.tab_buttons["tilt"] = make_button(tab_bar, "Tilt Processing", lambda: self._show_tab("tilt"), fg=FG_WARN)
        self.tab_buttons["tilt"].pack(side=tk.LEFT, padx=(0, 4))

        self.tab_buttons["roi"] = make_button(tab_bar, "ROI Visualization", lambda: self._show_tab("roi"))
        self.tab_buttons["roi"].pack(side=tk.LEFT, padx=(0, 4))

        self.content_frame = tk.Frame(self, bg=BG_MAIN, padx=12, pady=8)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self._build_tilt_tab()
        self._build_roi_tab()
        self._show_tab("tilt")

    def _show_tab(self, name):
        for key, frame in self.tab_frames.items():
            frame.pack_forget()
            self.tab_buttons[key].config(fg=FG_MAIN)
        self.tab_frames[name].pack(fill=tk.BOTH, expand=True)
        self.tab_buttons[name].config(fg=FG_WARN)

    # ========== TAB 1: TILT PROCESSING ==========
    def _build_tilt_tab(self):
        tab = tk.Frame(self.content_frame, bg=BG_MAIN)
        self.tab_frames["tilt"] = tab

        wrapper = tk.Frame(tab, bg=BG_MAIN)
        wrapper.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(wrapper, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1, width=470)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left.pack_propagate(False)

        right = tk.Frame(wrapper, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_tilt_left(left)
        self._build_tilt_right(right)

    def _build_tilt_left(self, parent):
        make_label(parent, "BASE DATA DIRECTORY").pack(anchor="w", padx=10, pady=(10, 2))
        make_entry(parent, self.base_data_var).pack(anchor="w", padx=10, pady=(0, 8))

        make_label(parent, "MASKS ROOT DIRECTORY").pack(anchor="w", padx=10, pady=(2, 2))
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill=tk.X, padx=10, pady=(0, 8))
        make_entry(row, self.masks_root_var, width=46).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row, "Browse", self._browse_masks_root).pack(side=tk.LEFT)

        ctrl = tk.Frame(parent, bg=BG_PANEL)
        ctrl.pack(fill=tk.X, padx=10, pady=(0, 8))
        make_button(ctrl, "Refresh Folders", self._load_mask_subfolders).pack(side=tk.LEFT, padx=(0, 6))
        make_button(ctrl, "Select All", self._select_all).pack(side=tk.LEFT, padx=(0, 6))
        make_button(ctrl, "Clear", self._clear_selection).pack(side=tk.LEFT)

        make_label(parent, "AVAILABLE TOOL MASK FOLDERS").pack(anchor="w", padx=10, pady=(2, 2))

        list_frame = tk.Frame(parent, bg=BG_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        self.folder_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED,
            bg=BG_ENTRY, fg=FG_MAIN, selectbackground="#0d6f2b", selectforeground="#d8ffd8",
            highlightbackground=BORDER, highlightcolor=FG_WARN,
            relief=tk.FLAT, font=("Consolas", 11), activestyle="none",
        )
        self.folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.folder_listbox.bind("<<ListboxSelect>>", lambda _: self._update_selected_count())

        scroll = tk.Scrollbar(list_frame, command=self.folder_listbox.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.folder_listbox.config(yscrollcommand=scroll.set)

        tk.Label(parent, textvariable=self.selected_count_var,
                 bg=BG_PANEL, fg=FG_WARN, font=("Consolas", 11, "bold")).pack(anchor="w", padx=10, pady=(2, 10))

    def _build_tilt_right(self, parent):
        make_label(parent, "PROCESS CONTROL").pack(anchor="w", padx=10, pady=(10, 6))
        controls = tk.Frame(parent, bg=BG_PANEL)
        controls.pack(fill=tk.X, padx=10, pady=(0, 8))

        make_label(controls, "Max Workers").pack(side=tk.LEFT, padx=(0, 8))
        self.workers_spin = tk.Spinbox(
            controls, from_=1, to=16, textvariable=self.max_workers_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN,
            buttonbackground="#0b2a0f", relief=tk.FLAT, font=("Consolas", 11),
        )
        self.workers_spin.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(controls, text="(bounded for 16-core friendly processing)",
                 bg=BG_PANEL, fg=FG_DIM, font=("Consolas", 10)).pack(side=tk.LEFT)

        action_row = tk.Frame(parent, bg=BG_PANEL)
        action_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.run_btn = make_button(action_row, "Run Tilt Processing", self._run_processing, width=26, fg=FG_WARN)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = make_button(action_row, "Stop", self._stop_processing, width=10)
        self.stop_btn.pack(side=tk.LEFT)

        make_label(parent, "LIVE LOG").pack(anchor="w", padx=10, pady=(2, 2))
        self.log = make_log(parent, height=28)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._append_log("Matrix launcher ready. Select folders and run.\n")

    # ========== TAB 2: ROI VISUALIZATION ==========
    def _build_roi_tab(self):
        tab = tk.Frame(self.content_frame, bg=BG_MAIN)
        self.tab_frames["roi"] = tab

        wrapper = tk.Frame(tab, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        wrapper.pack(fill=tk.BOTH, expand=True)

        # tool selection
        row1 = tk.Frame(wrapper, bg=BG_PANEL)
        row1.pack(fill=tk.X, padx=10, pady=(10, 4))
        make_label(row1, "MASKS_TILTED FOLDER").pack(side=tk.LEFT, padx=(0, 8))
        self.roi_tilted_var = tk.StringVar(value=DEFAULT_MASKS_TILTED_DIR)
        make_entry(row1, self.roi_tilted_var, width=70).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Browse", self._browse_roi_tilted).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Load Tools", self._load_roi_tools).pack(side=tk.LEFT)

        row2 = tk.Frame(wrapper, bg=BG_PANEL)
        row2.pack(fill=tk.X, padx=10, pady=(4, 4))
        make_label(row2, "Tool").pack(side=tk.LEFT, padx=(0, 8))
        self.roi_tool_var = tk.StringVar()
        self.roi_tool_combo = tk.Listbox(
            row2, height=5, bg=BG_ENTRY, fg=FG_MAIN,
            selectbackground="#0d6f2b", selectforeground="#d8ffd8",
            highlightbackground=BORDER, relief=tk.FLAT, font=("Consolas", 11),
            activestyle="none", exportselection=False,
        )
        self.roi_tool_combo.pack(side=tk.LEFT, padx=(0, 8), fill=tk.Y)
        self.roi_tool_combo.bind("<<ListboxSelect>>", self._on_roi_tool_selected)

        frame_col = tk.Frame(row2, bg=BG_PANEL)
        frame_col.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

        make_label(frame_col, "Frame Index (blank=random)").pack(anchor="w")
        self.roi_frame_var = tk.StringVar(value="")
        make_entry(frame_col, self.roi_frame_var, width=10).pack(anchor="w")

        make_label(frame_col, "Caption Override (blank=auto)").pack(anchor="w", pady=(6, 0))
        self.roi_caption_var = tk.StringVar(value="")
        make_entry(frame_col, self.roi_caption_var, width=40).pack(anchor="w")

        row3 = tk.Frame(wrapper, bg=BG_PANEL)
        row3.pack(fill=tk.X, padx=10, pady=(6, 4))
        make_button(row3, "Generate ROI Figure", self._generate_roi_figure, width=26, fg=FG_WARN).pack(side=tk.LEFT, padx=(0, 8))

        make_label(wrapper, "ROI LOG").pack(anchor="w", padx=10, pady=(6, 2))
        self.roi_log = make_log(wrapper, height=22)
        self.roi_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._roi_log("ROI tab ready. Load tools from masks_tilted directory.\n")

    # ================================================================
    # TILT TAB LOGIC
    # ================================================================
    def _animate_status(self):
        cur = self.status_label.cget("text")
        self.status_label.config(text="[ RUNNING ]" if cur == "[ READY ]" else "[ READY ]")
        self.after(700, self._animate_status)

    def _append_log(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _append_log_ts(self, text):
        self.after(0, lambda: self._append_log(text))

    def _browse_masks_root(self):
        cur = self.masks_root_var.get().strip() or DEFAULT_MASKS_DIR
        folder = filedialog.askdirectory(initialdir=cur, title="Select masks root folder")
        if folder:
            self.masks_root_var.set(folder)
            self._load_mask_subfolders()

    def _load_mask_subfolders(self):
        masks_root = self.masks_root_var.get().strip()
        self.folder_listbox.delete(0, tk.END)
        self.folder_names = []
        if not os.path.isdir(masks_root):
            self._append_log(f"Invalid masks root: {masks_root}\n")
            self._update_selected_count()
            return
        subdirs = sorted(
            name for name in os.listdir(masks_root)
            if os.path.isdir(os.path.join(masks_root, name))
        )
        for name in subdirs:
            self.folder_listbox.insert(tk.END, name)
        self.folder_names = subdirs
        self._append_log(f"Loaded {len(subdirs)} folders from {masks_root}\n")
        self._update_selected_count()

    def _select_all(self):
        self.folder_listbox.select_set(0, tk.END)
        self._update_selected_count()

    def _clear_selection(self):
        self.folder_listbox.selection_clear(0, tk.END)
        self._update_selected_count()

    def _update_selected_count(self):
        self.selected_count_var.set(f"Selected: {len(self.folder_listbox.curselection())}")

    def _get_selected_folder_paths(self):
        masks_root = self.masks_root_var.get().strip()
        return [os.path.join(masks_root, self.folder_names[i]) for i in self.folder_listbox.curselection()]

    def _set_running(self, running):
        self.run_btn.config(state=tk.DISABLED if running else tk.NORMAL)

    def _run_processing(self):
        base = self.base_data_var.get().strip()
        masks_root = self.masks_root_var.get().strip()
        selected = self._get_selected_folder_paths()

        if not os.path.isdir(base):
            messagebox.showerror("Path", f"Base DATA dir missing:\n{base}")
            return
        if not os.path.isdir(masks_root):
            messagebox.showerror("Path", f"Masks root missing:\n{masks_root}")
            return
        if not selected:
            messagebox.showwarning("Selection", "Select one or more tool folders.")
            return
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        workers = max(1, int(self.max_workers_var.get()))
        cmd = [sys.executable, PROCESS_SCRIPT, "--base-data-dir", base, "--max-workers", str(workers)]
        for f in selected:
            cmd.extend(["--mask-folder", f])

        self._set_running(True)
        self._append_log("\n" + "=" * 90 + "\n")
        self._append_log("Executing:\n" + " ".join(cmd) + "\n\n")
        threading.Thread(target=self._subprocess_worker, args=(cmd,), daemon=True).start()

    def _subprocess_worker(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd, cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            if self.process.stdout:
                for line in self.process.stdout:
                    self._append_log_ts(line)
            ec = self.process.wait()
            self._append_log_ts(f"\nProcess finished with exit code {ec}.\n")
        except Exception as exc:
            self._append_log_ts(f"\nFailed: {exc}\n")
        finally:
            self.process = None
            self.after(0, lambda: self._set_running(False))

    def _stop_processing(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self._append_log("Termination signal sent.\n")
            except Exception as exc:
                self._append_log(f"Stop failed: {exc}\n")
        else:
            self._append_log("No active process.\n")

    # ================================================================
    # ROI TAB LOGIC
    # ================================================================
    def _roi_log(self, text):
        self.roi_log.insert(tk.END, text)
        self.roi_log.see(tk.END)

    def _browse_roi_tilted(self):
        cur = self.roi_tilted_var.get().strip() or DEFAULT_MASKS_TILTED_DIR
        folder = filedialog.askdirectory(initialdir=cur, title="Select masks_tilted root")
        if folder:
            self.roi_tilted_var.set(folder)

    def _load_roi_tools(self):
        tilted_root = self.roi_tilted_var.get().strip()
        self.roi_tool_combo.delete(0, tk.END)
        if not os.path.isdir(tilted_root):
            self._roi_log(f"Not a valid directory: {tilted_root}\n")
            return
        subdirs = sorted(
            name for name in os.listdir(tilted_root)
            if os.path.isdir(os.path.join(tilted_root, name))
        )
        for name in subdirs:
            self.roi_tool_combo.insert(tk.END, name)
        self._roi_log(f"Loaded {len(subdirs)} tool folders from masks_tilted.\n")

    def _on_roi_tool_selected(self, _evt):
        sel = self.roi_tool_combo.curselection()
        if sel:
            self.roi_tool_var.set(self.roi_tool_combo.get(sel[0]))

    def _generate_roi_figure(self):
        sel = self.roi_tool_combo.curselection()
        if not sel:
            messagebox.showwarning("Selection", "Select a tool folder first.")
            return
        tool_folder_name = self.roi_tool_combo.get(sel[0])
        tilted_root = self.roi_tilted_var.get().strip()
        tool_dir = os.path.join(tilted_root, tool_folder_name)
        info_dir = os.path.join(tool_dir, "information")

        if not os.path.isdir(tool_dir):
            self._roi_log(f"Tool directory missing: {tool_dir}\n")
            return

        # Load master mask
        master_files = [
            f for f in os.listdir(info_dir)
            if "MASTER_MASK" in f.upper() and f.lower().endswith(".png")
        ] if os.path.isdir(info_dir) else []

        if not master_files:
            self._roi_log(f"No master mask PNG found in {info_dir}\n")
            return

        master_path = os.path.join(info_dir, master_files[0])
        master = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
        if master is None:
            self._roi_log(f"Cannot read master mask: {master_path}\n")
            return

        _, master_bin = cv2.threshold(master, 127, 255, cv2.THRESH_BINARY)

        # Load metadata for centerline
        meta_files = [f for f in os.listdir(info_dir) if f.endswith("_tilt_metadata.json")]
        tool_type = ""
        if meta_files:
            import json
            with open(os.path.join(info_dir, meta_files[0]), "r") as mf:
                meta = json.load(mf)
            tool_type = meta.get("tool_type", "")

        # Get tilted frame PNGs (exclude information folder)
        frame_files = sorted(
            f for f in os.listdir(tool_dir)
            if f.lower().endswith(".png") and os.path.isfile(os.path.join(tool_dir, f))
        )
        if not frame_files:
            self._roi_log("No tilted PNG frames found.\n")
            return

        # Pick frame
        frame_idx_str = self.roi_frame_var.get().strip()
        if frame_idx_str.isdigit():
            frame_idx = int(frame_idx_str)
            frame_idx = max(0, min(frame_idx, len(frame_files) - 1))
        else:
            frame_idx = random.randint(0, len(frame_files) - 1)

        frame_path = os.path.join(tool_dir, frame_files[frame_idx])
        frame_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame_img is None:
            self._roi_log(f"Cannot read frame: {frame_path}\n")
            return

        _, frame_bin = cv2.threshold(frame_img, 127, 255, cv2.THRESH_BINARY)

        # Compute centerline on master
        h, w = master_bin.shape
        from build_master_masks_all_two_edge_tools import get_boundaries, select_widest_rows, fit_lines
        ys, lx, rx = get_boundaries(master_bin)
        if ys is None:
            self._roi_log("Cannot compute boundaries on master mask.\n")
            return
        ys_f, lx_f, rx_f = select_widest_rows(ys, lx, rx)
        ll, lr, lc = fit_lines(ys_f, lx_f, rx_f)

        # ROI geometry
        ys_mm = np.where(master_bin.any(axis=1))[0]
        if len(ys_mm) == 0:
            self._roi_log("Master mask appears empty.\n")
            return
        bottom_y = int(ys_mm[-1])
        white_cols = np.where(master_bin.any(axis=0))[0]
        if len(white_cols) < 2:
            self._roi_log("Master mask too narrow.\n")
            return
        mask_width = int(white_cols[-1] - white_cols[0])
        roi_height = int(round(0.45 * mask_width))
        roi_top = max(0, bottom_y - roi_height)
        roi_bottom = bottom_y
        roi_left = int(white_cols[0])
        roi_right = int(white_cols[-1])

        m_c, b_c = lc
        center_x = int(round(m_c * ((roi_top + roi_bottom) / 2.0) + b_c))

        # Build figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MplRect
        from matplotlib.lines import Line2D

        vis = cv2.cvtColor(frame_bin, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0

        alpha = 0.35
        blue_mask = np.zeros_like(vis)
        blue_mask[roi_top:roi_bottom, roi_left:center_x] = [0.0, 0.0, 1.0]
        red_mask = np.zeros_like(vis)
        red_mask[roi_top:roi_bottom, center_x:roi_right] = [1.0, 0.0, 0.0]

        bm = blue_mask.sum(axis=2) > 0
        rm = red_mask.sum(axis=2) > 0
        vis[bm] = vis[bm] * (1 - alpha) + blue_mask[bm] * alpha
        vis[rm] = vis[rm] * (1 - alpha) + red_mask[rm] * alpha

        caption = self.roi_caption_var.get().strip()
        if not caption:
            ttype = (tool_type or "Tool").capitalize()
            match = re.search(r"(tool\d+)", tool_folder_name, re.IGNORECASE)
            tid = match.group(1) if match else tool_folder_name
            tid_num = re.sub(r"[^0-9]", "", tid)
            caption = f"{ttype} Tool ({tid_num})"

        fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
        ax.imshow(vis, origin="upper")
        rect = MplRect((roi_left, roi_top), roi_right - roi_left, roi_bottom - roi_top,
                        linewidth=2.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.plot([center_x, center_x], [roi_top, roi_bottom],
                color="yellow", linewidth=2.5)

        legend_handles = [
            MplRect((0, 0), 1, 1, facecolor="blue", alpha=0.5, label="Left ROI"),
            MplRect((0, 0), 1, 1, facecolor="red", alpha=0.5, label="Right ROI"),
            Line2D([0], [0], color="yellow", linewidth=2.5, label="Centerline in ROI"),
            MplRect((0, 0), 1, 1, edgecolor="lime", facecolor="none", linewidth=2, label="ROI Box"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=11)
        ax.set_title(caption, fontsize=18)
        ax.set_axis_off()
        fig.tight_layout(pad=0.15)

        out_path = os.path.join(info_dir, f"{tid}_roi_visualization.png")
        os.makedirs(info_dir, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        self._roi_log(
            f"ROI figure saved: {out_path}\n"
            f"  Frame used: {frame_files[frame_idx]} (index {frame_idx})\n"
            f"  ROI height: {roi_height}px (0.45 * mask_width={mask_width}px)\n"
            f"  Caption: {caption}\n"
        )


if __name__ == "__main__":
    # Add script dir to path so ROI tab can import processing functions
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    app = MatrixPerspectiveGUI()
    app.mainloop()
