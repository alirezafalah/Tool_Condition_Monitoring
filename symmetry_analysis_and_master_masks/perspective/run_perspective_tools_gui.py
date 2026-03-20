import json
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

from find_optimal_offset import OffsetAnalysisConfig, run_optimal_offset_analysis_for_tool, run_symmetry_summary, run_custom_summary_graph

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "build_master_masks_all_two_edge_tools.py")

DEFAULT_BASE_DATA_DIR = (
    r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
)
DEFAULT_MASKS_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, "masks")
DEFAULT_MASKS_TILTED_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, "masks_tilted")
DEFAULT_SYMMETRY_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, "symmetry")
DEFAULT_TOOLS_METADATA = os.path.join(DEFAULT_BASE_DATA_DIR, "tools_metadata.csv")


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
        self.roi_preview_canvas = None
        self.roi_preview_fig = None
        self.roi_current_context = None
        self.roi_current_geo = None
        self.roi_edit_target_var = tk.StringVar(value="centerline")
        self.roi_step_var = tk.IntVar(value=1)

        self.offset_running = False
        self.offset_cancel_requested = False
        self.offset_tilted_var = tk.StringVar(value=DEFAULT_MASKS_TILTED_DIR)
        self.offset_mode_var = tk.StringVar(value="search_offset")
        self.offset_num_frames_var = tk.IntVar(value=90)
        self.offset_roi_height_var = tk.IntVar(value=200)
        self.offset_use_metadata_roi_var = tk.BooleanVar(value=True)
        self.offset_min_var = tk.IntVar(value=176)
        self.offset_max_var = tk.IntVar(value=186)
        self.offset_search_num_regions_var = tk.IntVar(value=2)
        self.offset_range_a_start_var = tk.IntVar(value=0)
        self.offset_range_a_end_var = tk.IntVar(value=89)
        self.offset_range_b_start_var = tk.IntVar(value=182)
        self.offset_range_b_end_var = tk.IntVar(value=271)
        self.offset_regions_text_var = tk.StringVar(value="0-89,182-271")
        self.offset_fmt_png_var = tk.BooleanVar(value=True)
        self.offset_fmt_svg_var = tk.BooleanVar(value=False)
        self.offset_fmt_pdf_var = tk.BooleanVar(value=False)
        self.offset_stack_overlay_abs_diff_var = tk.BooleanVar(value=False)
        self.offset_title_font_var = tk.IntVar(value=14)
        self.offset_axis_font_var = tk.IntVar(value=12)
        self.offset_tick_font_var = tk.IntVar(value=10)
        self.offset_legend_font_var = tk.IntVar(value=10)
        self.offset_include_caption_var = tk.BooleanVar(value=True)
        self.offset_manual_legend_var = tk.BooleanVar(value=False)
        self.offset_legend_a_start_var = tk.IntVar(value=0)
        self.offset_legend_a_end_var = tk.IntVar(value=90)
        self.offset_legend_b_start_var = tk.IntVar(value=180)
        self.offset_legend_b_end_var = tk.IntVar(value=270)
        self.offset_legend_ranges_text_var = tk.StringVar(value="0-90,180-270")

        self.summary_running = False
        
        # Custom Graph tab variables
        self.custom_graph_running = False
        self.custom_graph_sym_var = tk.StringVar(value=DEFAULT_SYMMETRY_DIR)
        self.custom_graph_meta_var = tk.StringVar(value=DEFAULT_TOOLS_METADATA)
        self.custom_graph_num_labels_var = tk.IntVar(value=4)
        self.custom_graph_title_var = tk.StringVar(value="Custom Summary: Mean Absolute Difference")
        self.custom_graph_show_title_var = tk.BooleanVar(value=True)
        self.custom_graph_label_vars = []  # Will hold (name_var, color_var, tools_listbox) tuples
        self.custom_graph_all_tools_listbox = None
        self.custom_graph_labels_frame = None
        self.custom_graph_label_vars = []
        self.custom_graph_tool_assignments = {}  # {label_idx: [tool_ids...]}
        self.custom_graph_available_tools_listbox = None  # Central tool pool
        self.custom_graph_dragging_tools = []  # Currently dragged tools (multi-select)
        self.custom_graph_fmt_png_var = tk.BooleanVar(value=True)
        self.custom_graph_fmt_svg_var = tk.BooleanVar(value=False)
        self.custom_graph_fmt_pdf_var = tk.BooleanVar(value=False)
        self.custom_graph_title_font_var = tk.IntVar(value=14)
        self.custom_graph_axis_font_var = tk.IntVar(value=12)
        self.custom_graph_tick_font_var = tk.IntVar(value=10)
        self.custom_graph_legend_font_var = tk.IntVar(value=10)
        self.custom_graph_show_threshold_var = tk.BooleanVar(value=False)
        self.custom_graph_threshold_var = tk.DoubleVar(value=0.0)

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

        self.tab_buttons["offset"] = make_button(tab_bar, "Optimal Offset", lambda: self._show_tab("offset"))
        self.tab_buttons["offset"].pack(side=tk.LEFT, padx=(0, 4))

        self.tab_buttons["summary"] = make_button(tab_bar, "Symmetry Summary", lambda: self._show_tab("summary"))
        self.tab_buttons["summary"].pack(side=tk.LEFT, padx=(0, 4))

        self.tab_buttons["custom"] = make_button(tab_bar, "Custom Summary", lambda: self._show_tab("custom"))
        self.tab_buttons["custom"].pack(side=tk.LEFT, padx=(0, 4))

        self.content_frame = tk.Frame(self, bg=BG_MAIN, padx=12, pady=8)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self._build_tilt_tab()
        self._build_roi_tab()
        self._build_offset_tab()
        self._build_summary_tab()
        self._build_custom_graph_tab()
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

        make_label(frame_col, "Caption Override (blank=auto, '-'=none)").pack(anchor="w", pady=(6, 0))
        self.roi_caption_var = tk.StringVar(value="")
        make_entry(frame_col, self.roi_caption_var, width=40).pack(anchor="w")

        row3 = tk.Frame(wrapper, bg=BG_PANEL)
        row3.pack(fill=tk.X, padx=10, pady=(6, 4))
        make_button(row3, "Preview ROI", self._preview_roi, width=18, fg=FG_WARN).pack(side=tk.LEFT, padx=(0, 8))
        make_button(row3, "Save ROI + Metadata", self._save_roi_and_metadata, width=24).pack(side=tk.LEFT)

        row4 = tk.Frame(wrapper, bg=BG_PANEL)
        row4.pack(fill=tk.X, padx=10, pady=(2, 6))
        make_label(row4, "Manual Edit").pack(side=tk.LEFT, padx=(0, 8))
        self.roi_target_menu = tk.OptionMenu(
            row4,
            self.roi_edit_target_var,
            "top", "bottom", "left", "right", "centerline",
            command=lambda _v: self._update_adjust_buttons_state(),
        )
        self.roi_target_menu.config(
            bg=BG_ENTRY,
            fg=FG_MAIN,
            activebackground=BG_BTN_ACTIVE,
            activeforeground="#dcffdc",
            relief=tk.FLAT,
            highlightbackground=BORDER,
            highlightthickness=1,
            font=("Consolas", 10),
        )
        self.roi_target_menu["menu"].config(bg=BG_ENTRY, fg=FG_MAIN, font=("Consolas", 10))
        self.roi_target_menu.pack(side=tk.LEFT, padx=(0, 10))

        make_label(row4, "Step(px)").pack(side=tk.LEFT, padx=(0, 6))
        self.roi_step_spin = tk.Spinbox(
            row4,
            from_=1,
            to=50,
            textvariable=self.roi_step_var,
            width=5,
            bg=BG_ENTRY,
            fg=FG_MAIN,
            insertbackground=FG_MAIN,
            buttonbackground="#0b2a0f",
            relief=tk.FLAT,
            font=("Consolas", 10),
        )
        self.roi_step_spin.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_move_left = make_button(row4, "<- Left", lambda: self._move_selected_edge(dx=-self._get_move_step(), dy=0), width=9)
        self.btn_move_left.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_move_right = make_button(row4, "Right ->", lambda: self._move_selected_edge(dx=self._get_move_step(), dy=0), width=9)
        self.btn_move_right.pack(side=tk.LEFT, padx=(0, 12))
        self.btn_move_up = make_button(row4, "Up", lambda: self._move_selected_edge(dx=0, dy=-self._get_move_step()), width=7)
        self.btn_move_up.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_move_down = make_button(row4, "Down", lambda: self._move_selected_edge(dx=0, dy=self._get_move_step()), width=7)
        self.btn_move_down.pack(side=tk.LEFT)

        self._update_adjust_buttons_state()

        make_label(wrapper, "ROI PREVIEW").pack(anchor="w", padx=10, pady=(6, 2))
        self.roi_preview_frame = tk.Frame(
            wrapper,
            bg=BG_ENTRY,
            highlightbackground=BORDER,
            highlightthickness=1,
            height=360,
        )
        self.roi_preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.roi_preview_frame.pack_propagate(False)
        self.roi_preview_frame.bind("<Configure>", self._on_roi_preview_resize)

        make_label(wrapper, "ROI LOG").pack(anchor="w", padx=10, pady=(6, 2))
        self.roi_log = make_log(wrapper, height=8)
        self.roi_log.pack(fill=tk.X, expand=False, padx=10, pady=(0, 10))
        self._roi_log("ROI tab ready. Load tools from masks_tilted directory.\n")

    # ========== TAB 3: OPTIMAL OFFSET ===========
    def _build_offset_tab(self):
        tab = tk.Frame(self.content_frame, bg=BG_MAIN)
        self.tab_frames["offset"] = tab

        wrapper = tk.Frame(tab, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        wrapper.pack(fill=tk.BOTH, expand=True)

        row1 = tk.Frame(wrapper, bg=BG_PANEL)
        row1.pack(fill=tk.X, padx=10, pady=(10, 4))
        make_label(row1, "MASKS_TILTED FOLDER").pack(side=tk.LEFT, padx=(0, 8))
        make_entry(row1, self.offset_tilted_var, width=68).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Browse", self._browse_offset_tilted).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Load Tools", self._load_offset_tools).pack(side=tk.LEFT)

        row2 = tk.Frame(wrapper, bg=BG_PANEL)
        row2.pack(fill=tk.X, padx=10, pady=(4, 4))

        list_col = tk.Frame(row2, bg=BG_PANEL)
        list_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        make_label(list_col, "Tools (select one or more)").pack(anchor="w")
        self.offset_tool_listbox = tk.Listbox(
            list_col,
            selectmode=tk.EXTENDED,
            height=8,
            bg=BG_ENTRY,
            fg=FG_MAIN,
            selectbackground="#0d6f2b",
            selectforeground="#d8ffd8",
            highlightbackground=BORDER,
            relief=tk.FLAT,
            font=("Consolas", 11),
            activestyle="none",
            exportselection=False,
            width=38,
        )
        self.offset_tool_listbox.pack(fill=tk.BOTH, expand=True)
        self.offset_tool_listbox.bind("<<ListboxSelect>>", self._on_offset_tool_selected)

        opt_col = tk.Frame(row2, bg=BG_PANEL)
        opt_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.offset_search_widgets = []
        self.offset_fixed_widgets = []
        self.offset_legend_widgets = []

        line_mode = tk.Frame(opt_col, bg=BG_PANEL)
        line_mode.pack(fill=tk.X, pady=(0, 6))
        make_label(line_mode, "Mode").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_mode_menu = tk.OptionMenu(
            line_mode,
            self.offset_mode_var,
            "search_offset",
            "fixed_ranges",
            command=lambda _v: self._update_offset_mode_widgets(),
        )
        self.offset_mode_menu.config(
            bg=BG_ENTRY,
            fg=FG_MAIN,
            activebackground=BG_BTN_ACTIVE,
            activeforeground="#dcffdc",
            relief=tk.FLAT,
            highlightbackground=BORDER,
            highlightthickness=1,
            font=("Consolas", 10),
        )
        self.offset_mode_menu["menu"].config(bg=BG_ENTRY, fg=FG_MAIN, font=("Consolas", 10))
        self.offset_mode_menu.pack(side=tk.LEFT, padx=(0, 12))

        tk.Checkbutton(
            line_mode,
            text="Use ROI Height from metadata JSON",
            variable=self.offset_use_metadata_roi_var,
            command=self._try_load_offset_roi_from_metadata,
            bg=BG_PANEL,
            fg=FG_MAIN,
            selectcolor=BG_ENTRY,
            activebackground=BG_PANEL,
            activeforeground=FG_MAIN,
            font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))

        line_a = tk.Frame(opt_col, bg=BG_PANEL)
        line_a.pack(fill=tk.X, pady=(0, 6))
        make_label(line_a, "Frames").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_frames_spin = tk.Spinbox(
            line_a, from_=1, to=1000, textvariable=self.offset_num_frames_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_frames_spin.pack(side=tk.LEFT, padx=(0, 10))
        make_label(line_a, "ROI Height(px)").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_roi_spin = tk.Spinbox(
            line_a, from_=1, to=2000, textvariable=self.offset_roi_height_var, width=7,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_roi_spin.pack(side=tk.LEFT)

        self.offset_search_widgets.extend([self.offset_frames_spin])

        line_b = tk.Frame(opt_col, bg=BG_PANEL)
        line_b.pack(fill=tk.X, pady=(0, 6))
        make_label(line_b, "Offset Min").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_min_spin = tk.Spinbox(
            line_b, from_=0, to=360, textvariable=self.offset_min_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_min_spin.pack(side=tk.LEFT, padx=(0, 10))
        make_label(line_b, "Offset Max").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_max_spin = tk.Spinbox(
            line_b, from_=0, to=360, textvariable=self.offset_max_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_max_spin.pack(side=tk.LEFT, padx=(0, 10))

        make_label(line_b, "Num Regions").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_search_num_regions_spin = tk.Spinbox(
            line_b, from_=2, to=10, textvariable=self.offset_search_num_regions_var, width=4,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_search_num_regions_spin.pack(side=tk.LEFT)

        self.offset_search_widgets.extend([self.offset_min_spin, self.offset_max_spin, self.offset_search_num_regions_spin])

        line_fixed = tk.Frame(opt_col, bg=BG_PANEL)
        line_fixed.pack(fill=tk.X, pady=(0, 6))
        make_label(line_fixed, "Range A (start-end)").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_range_a_start_spin = tk.Spinbox(
            line_fixed, from_=0, to=5000, textvariable=self.offset_range_a_start_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_range_a_start_spin.pack(side=tk.LEFT, padx=(0, 4))
        self.offset_range_a_end_spin = tk.Spinbox(
            line_fixed, from_=0, to=5000, textvariable=self.offset_range_a_end_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_range_a_end_spin.pack(side=tk.LEFT, padx=(0, 10))

        make_label(line_fixed, "Range B (start-end)").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_range_b_start_spin = tk.Spinbox(
            line_fixed, from_=0, to=5000, textvariable=self.offset_range_b_start_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_range_b_start_spin.pack(side=tk.LEFT, padx=(0, 4))
        self.offset_range_b_end_spin = tk.Spinbox(
            line_fixed, from_=0, to=5000, textvariable=self.offset_range_b_end_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_range_b_end_spin.pack(side=tk.LEFT)

        self.offset_fixed_widgets.extend(
            [
                self.offset_range_a_start_spin,
                self.offset_range_a_end_spin,
                self.offset_range_b_start_spin,
                self.offset_range_b_end_spin,
            ]
        )

        line_fixed_text = tk.Frame(opt_col, bg=BG_PANEL)
        line_fixed_text.pack(fill=tk.X, pady=(0, 6))
        make_label(line_fixed_text, "Regions List (e.g. 0-60,120-180,240-300)").pack(side=tk.LEFT, padx=(0, 6))
        self.offset_regions_entry = make_entry(line_fixed_text, self.offset_regions_text_var, width=44)
        self.offset_regions_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.offset_fixed_widgets.append(self.offset_regions_entry)

        line_c = tk.Frame(opt_col, bg=BG_PANEL)
        line_c.pack(fill=tk.X, pady=(0, 6))
        make_label(line_c, "Output Formats").pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            line_c, text="PNG", variable=self.offset_fmt_png_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(
            line_c, text="SVG", variable=self.offset_fmt_svg_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(
            line_c, text="PDF", variable=self.offset_fmt_pdf_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        line_c2 = tk.Frame(opt_col, bg=BG_PANEL)
        line_c2.pack(fill=tk.X, pady=(0, 6))
        tk.Checkbutton(
            line_c2,
            text="Create stacked figure (Overlay top + AbsDiff bottom)",
            variable=self.offset_stack_overlay_abs_diff_var,
            bg=BG_PANEL,
            fg=FG_MAIN,
            selectcolor=BG_ENTRY,
            activebackground=BG_PANEL,
            activeforeground=FG_MAIN,
            font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        line_d = tk.Frame(opt_col, bg=BG_PANEL)
        line_d.pack(fill=tk.X, pady=(0, 6))
        make_label(line_d, "Title Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            line_d, from_=8, to=48, textvariable=self.offset_title_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))
        make_label(line_d, "Axis Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            line_d, from_=8, to=48, textvariable=self.offset_axis_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))
        make_label(line_d, "Tick Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            line_d, from_=6, to=36, textvariable=self.offset_tick_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))
        make_label(line_d, "Legend Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            line_d, from_=6, to=48, textvariable=self.offset_legend_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        line_e = tk.Frame(opt_col, bg=BG_PANEL)
        line_e.pack(fill=tk.X, pady=(0, 4))
        tk.Checkbutton(
            line_e,
            text="Include Top Caption",
            variable=self.offset_include_caption_var,
            bg=BG_PANEL,
            fg=FG_MAIN,
            selectcolor=BG_ENTRY,
            activebackground=BG_PANEL,
            activeforeground=FG_MAIN,
            font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.offset_manual_legend_chk = tk.Checkbutton(
            line_e,
            text="Manual Legend Degree Ranges",
            variable=self.offset_manual_legend_var,
            command=self._update_offset_legend_widgets,
            bg=BG_PANEL,
            fg=FG_MAIN,
            selectcolor=BG_ENTRY,
            activebackground=BG_PANEL,
            activeforeground=FG_MAIN,
            font=("Consolas", 10),
        )
        self.offset_manual_legend_chk.pack(side=tk.LEFT)

        line_legend = tk.Frame(opt_col, bg=BG_PANEL)
        line_legend.pack(fill=tk.X, pady=(0, 6))
        make_label(line_legend, "Legend A (start-end deg)").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_legend_a_start_spin = tk.Spinbox(
            line_legend, from_=-360, to=720, textvariable=self.offset_legend_a_start_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_legend_a_start_spin.pack(side=tk.LEFT, padx=(0, 4))
        self.offset_legend_a_end_spin = tk.Spinbox(
            line_legend, from_=-360, to=720, textvariable=self.offset_legend_a_end_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_legend_a_end_spin.pack(side=tk.LEFT, padx=(0, 10))

        make_label(line_legend, "Legend B (start-end deg)").pack(side=tk.LEFT, padx=(0, 4))
        self.offset_legend_b_start_spin = tk.Spinbox(
            line_legend, from_=-360, to=720, textvariable=self.offset_legend_b_start_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_legend_b_start_spin.pack(side=tk.LEFT, padx=(0, 4))
        self.offset_legend_b_end_spin = tk.Spinbox(
            line_legend, from_=-360, to=720, textvariable=self.offset_legend_b_end_var, width=6,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        )
        self.offset_legend_b_end_spin.pack(side=tk.LEFT)

        self.offset_legend_widgets.extend(
            [
                self.offset_legend_a_start_spin,
                self.offset_legend_a_end_spin,
                self.offset_legend_b_start_spin,
                self.offset_legend_b_end_spin,
            ]
        )

        line_legend_text = tk.Frame(opt_col, bg=BG_PANEL)
        line_legend_text.pack(fill=tk.X, pady=(0, 6))
        make_label(line_legend_text, "Legend Ranges List (optional)").pack(side=tk.LEFT, padx=(0, 6))
        self.offset_legend_ranges_entry = make_entry(line_legend_text, self.offset_legend_ranges_text_var, width=38)
        self.offset_legend_ranges_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.offset_legend_widgets.append(self.offset_legend_ranges_entry)

        row3 = tk.Frame(wrapper, bg=BG_PANEL)
        row3.pack(fill=tk.X, padx=10, pady=(6, 4))
        self.offset_run_btn = make_button(
            row3,
            "Run Optimal Offset Analysis",
            self._run_offset_analysis,
            width=30,
            fg=FG_WARN,
        )
        self.offset_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.offset_stop_btn = make_button(row3, "Stop", self._stop_offset_analysis, width=10)
        self.offset_stop_btn.pack(side=tk.LEFT)

        make_label(wrapper, "OPTIMAL OFFSET LOG").pack(anchor="w", padx=10, pady=(6, 2))
        self.offset_log = make_log(wrapper, height=14)
        self.offset_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._offset_log("Optimal Offset tab ready. Load tools from masks_tilted directory.\n")
        self._update_offset_mode_widgets()
        self._update_offset_legend_widgets()

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
    # OPTIMAL OFFSET TAB LOGIC
    # ================================================================
    def _offset_log(self, text):
        self.offset_log.insert(tk.END, text)
        self.offset_log.see(tk.END)

    def _offset_log_ts(self, text):
        self.after(0, lambda: self._offset_log(text))

    def _browse_offset_tilted(self):
        cur = self.offset_tilted_var.get().strip() or DEFAULT_MASKS_TILTED_DIR
        folder = filedialog.askdirectory(initialdir=cur, title="Select masks_tilted root")
        if folder:
            self.offset_tilted_var.set(folder)

    def _load_offset_tools(self):
        tilted_root = self.offset_tilted_var.get().strip()
        self.offset_tool_listbox.delete(0, tk.END)
        if not os.path.isdir(tilted_root):
            self._offset_log(f"Not a valid directory: {tilted_root}\n")
            return

        subdirs = sorted(
            name for name in os.listdir(tilted_root)
            if os.path.isdir(os.path.join(tilted_root, name))
        )
        for name in subdirs:
            self.offset_tool_listbox.insert(tk.END, name)
        self._offset_log(f"Loaded {len(subdirs)} tool folders from masks_tilted.\n")
        if subdirs:
            self.offset_tool_listbox.selection_set(0)
            if self.offset_use_metadata_roi_var.get():
                self._try_load_offset_roi_from_metadata()

    def _on_offset_tool_selected(self, _evt):
        if self.offset_use_metadata_roi_var.get():
            self._try_load_offset_roi_from_metadata()

    def _try_load_offset_roi_from_metadata(self):
        selected = self._get_selected_offset_tool_paths()
        if not selected:
            return

        tool_dir = selected[0]
        info_dir = os.path.join(tool_dir, "information")
        if not os.path.isdir(info_dir):
            return

        meta_files = sorted(
            f for f in os.listdir(info_dir)
            if f.endswith("_tilt_metadata.json") and os.path.isfile(os.path.join(info_dir, f))
        )
        for name in meta_files:
            path = os.path.join(info_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                roi_height = data.get("roi_height_px", data.get("roi_height"))
                if roi_height is None:
                    continue
                roi_height = int(roi_height)
                if roi_height > 0:
                    self.offset_roi_height_var.set(roi_height)
                    self._offset_log(
                        f"Loaded ROI Height={roi_height}px from metadata for {os.path.basename(tool_dir)}.\n"
                    )
                    return
            except Exception:
                continue

    def _update_offset_mode_widgets(self):
        mode = self.offset_mode_var.get().strip()
        search_state = tk.NORMAL if mode == "search_offset" else tk.DISABLED
        fixed_state = tk.NORMAL if mode == "fixed_ranges" else tk.DISABLED

        for widget in self.offset_search_widgets:
            widget.config(state=search_state)
        for widget in self.offset_fixed_widgets:
            widget.config(state=fixed_state)

        # Manual legend is available in BOTH modes.
        self.offset_manual_legend_chk.config(state=tk.NORMAL)
        self._update_offset_legend_widgets()

    def _update_offset_legend_widgets(self):
        state = tk.NORMAL if self.offset_manual_legend_var.get() else tk.DISABLED
        for widget in self.offset_legend_widgets:
            widget.config(state=state)

    def _set_offset_running(self, running):
        self.offset_running = running
        self.offset_run_btn.config(state=tk.DISABLED if running else tk.NORMAL)

    def _get_selected_output_formats(self):
        fmts = []
        if self.offset_fmt_png_var.get():
            fmts.append("png")
        if self.offset_fmt_svg_var.get():
            fmts.append("svg")
        if self.offset_fmt_pdf_var.get():
            fmts.append("pdf")
        return tuple(fmts)

    def _get_selected_offset_tool_paths(self):
        root = self.offset_tilted_var.get().strip()
        return [
            os.path.join(root, self.offset_tool_listbox.get(i))
            for i in self.offset_tool_listbox.curselection()
        ]

    @staticmethod
    def _parse_ranges_text(ranges_text):
        text = (ranges_text or "").strip()
        if not text:
            return []

        parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
        ranges = []
        for part in parts:
            m = re.match(r"^(-?\d+)\s*[-:]\s*(-?\d+)$", part)
            if not m:
                raise ValueError(f"Invalid range entry: '{part}'")
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                raise ValueError(f"Range end must be >= start: '{part}'")
            ranges.append((start, end))
        return ranges

    def _run_offset_analysis(self):
        if self.offset_running:
            messagebox.showwarning("Busy", "Optimal offset analysis is already running.")
            return

        tool_paths = self._get_selected_offset_tool_paths()
        if not tool_paths:
            messagebox.showwarning("Selection", "Select one or more tools to analyze.")
            return

        output_formats = self._get_selected_output_formats()
        if not output_formats:
            messagebox.showerror("Output Format", "Select at least one output format (PNG, SVG, or PDF).")
            return

        mode = self.offset_mode_var.get().strip()
        if mode not in {"search_offset", "fixed_ranges"}:
            messagebox.showerror("Mode", f"Unsupported mode: {mode}")
            return

        fixed_regions = []
        legend_ranges = []

        if mode == "search_offset":
            offset_min = int(self.offset_min_var.get())
            offset_max = int(self.offset_max_var.get())
            if offset_max < offset_min:
                messagebox.showerror("Range", "Offset Max must be >= Offset Min.")
                return
        else:
            try:
                fixed_regions = self._parse_ranges_text(self.offset_regions_text_var.get())
            except ValueError as exc:
                messagebox.showerror("Fixed Regions", str(exc))
                return

            if not fixed_regions:
                range_a_start = int(self.offset_range_a_start_var.get())
                range_a_end = int(self.offset_range_a_end_var.get())
                range_b_start = int(self.offset_range_b_start_var.get())
                range_b_end = int(self.offset_range_b_end_var.get())
                if range_a_end < range_a_start or range_b_end < range_b_start:
                    messagebox.showerror("Range", "Range end must be >= range start for both A and B.")
                    return
                fixed_regions = [(range_a_start, range_a_end), (range_b_start, range_b_end)]

            if len(fixed_regions) < 2:
                messagebox.showerror("Fixed Regions", "Provide at least two ranges for fixed mode.")
                return

            offset_min = int(self.offset_min_var.get())
            offset_max = int(self.offset_max_var.get())

        # Manual legend handling — available in both modes.
        if self.offset_manual_legend_var.get():
            try:
                legend_ranges = self._parse_ranges_text(self.offset_legend_ranges_text_var.get())
            except ValueError as exc:
                messagebox.showerror("Legend Ranges", str(exc))
                return

            if mode == "fixed_ranges" and fixed_regions:
                if legend_ranges and len(legend_ranges) != len(fixed_regions):
                    messagebox.showerror(
                        "Legend Ranges",
                        "Legend range count must match fixed region count, or leave it blank to use fallback values.",
                    )
                    return
                if len(fixed_regions) > 2 and not legend_ranges:
                    messagebox.showerror(
                        "Legend Ranges",
                        "For more than two regions in manual legend mode, provide Legend Ranges List.",
                    )
                    return

        cfg = OffsetAnalysisConfig(
            analysis_mode=mode,
            num_frames=max(1, int(self.offset_num_frames_var.get())),
            search_num_regions=max(2, int(self.offset_search_num_regions_var.get())),
            roi_height=max(1, int(self.offset_roi_height_var.get())),
            use_metadata_roi_height=bool(self.offset_use_metadata_roi_var.get()),
            offset_min=offset_min,
            offset_max=offset_max,
            range_a_start=int(self.offset_range_a_start_var.get()),
            range_a_end=int(self.offset_range_a_end_var.get()),
            range_b_start=int(self.offset_range_b_start_var.get()),
            range_b_end=int(self.offset_range_b_end_var.get()),
            region_ranges=tuple(fixed_regions),
            output_formats=output_formats,
            stack_overlay_abs_diff=bool(self.offset_stack_overlay_abs_diff_var.get()),
            title_font_size=max(1, int(self.offset_title_font_var.get())),
            axis_label_font_size=max(1, int(self.offset_axis_font_var.get())),
            tick_font_size=max(1, int(self.offset_tick_font_var.get())),
            legend_font_size=max(1, int(self.offset_legend_font_var.get())),
            include_top_caption=bool(self.offset_include_caption_var.get()),
            manual_legend_ranges=bool(self.offset_manual_legend_var.get()),
            legend_a_start_deg=int(self.offset_legend_a_start_var.get()),
            legend_a_end_deg=int(self.offset_legend_a_end_var.get()),
            legend_b_start_deg=int(self.offset_legend_b_start_var.get()),
            legend_b_end_deg=int(self.offset_legend_b_end_var.get()),
            legend_ranges=tuple(legend_ranges),
        )

        self.offset_cancel_requested = False
        self._set_offset_running(True)
        self._offset_log("\n" + "=" * 90 + "\n")
        if cfg.analysis_mode == "search_offset":
            self._offset_log(
                f"Running search mode for {len(tool_paths)} tool(s) | "
                f"frames={cfg.num_frames}, offsets={cfg.offset_min}-{cfg.offset_max}, "
                f"roi_height={'metadata' if cfg.use_metadata_roi_height else cfg.roi_height}\n"
            )
        else:
            self._offset_log(
                f"Running fixed-range mode for {len(tool_paths)} tool(s) | "
                f"regions={','.join(f'{a}-{b}' for a, b in cfg.region_ranges)}, "
                f"roi_height={'metadata' if cfg.use_metadata_roi_height else cfg.roi_height}\n"
            )

        threading.Thread(
            target=self._offset_analysis_worker,
            args=(tool_paths, cfg),
            daemon=True,
        ).start()

    def _offset_analysis_worker(self, tool_paths, cfg):
        try:
            for idx, tool_path in enumerate(tool_paths, start=1):
                if self.offset_cancel_requested:
                    self._offset_log_ts("Stop requested. Ending after current tool.\n")
                    break

                self._offset_log_ts("\n" + "-" * 70 + "\n")
                self._offset_log_ts(f"[{idx}/{len(tool_paths)}] Analyzing {os.path.basename(tool_path)}\n")

                try:
                    # Compute symmetry output directory for this tool.
                    _tid_match = re.search(r"(tool\d+)", os.path.basename(tool_path), re.IGNORECASE)
                    _tid = _tid_match.group(1).lower() if _tid_match else os.path.basename(tool_path)
                    _sym_dir = os.path.join(self.base_data_var.get().strip(), "symmetry", _tid)

                    result = run_optimal_offset_analysis_for_tool(
                        tool_path,
                        cfg,
                        log_fn=self._offset_log_ts,
                        symmetry_dir=_sym_dir,
                    )
                    if result.get("analysis_mode") == "search_offset":
                        self._offset_log_ts(
                            f"Optimal offset: {result['optimal_offset']} deg | "
                            f"frames {result['frame_range']} | ROI={result['roi_height_px']}px\n"
                            f"Saved in: {result['output_dir']}\n"
                        )
                    else:
                        self._offset_log_ts(
                            f"Fixed comparison complete | regions={'; '.join(result['internal_regions'])} | "
                            f"display={'; '.join(result['display_regions'])} deg | "
                            f"pairs={result['pair_count']} | mean_abs_diff={result['mean_abs_diff']:.2f} | ROI={result['roi_height_px']}px\n"
                            f"Saved in: {result['output_dir']}\n"
                        )
                except Exception as exc:
                    self._offset_log_ts(f"ERROR for {os.path.basename(tool_path)}: {exc}\n")
        finally:
            self.after(0, lambda: self._set_offset_running(False))

    def _stop_offset_analysis(self):
        if not self.offset_running:
            self._offset_log("No active optimal offset analysis.\n")
            return
        self.offset_cancel_requested = True
        self._offset_log("Stop requested. Waiting for current step to finish...\n")

    # ================================================================
    # TAB 4: SYMMETRY SUMMARY
    # ================================================================
    def _build_summary_tab(self):
        tab = tk.Frame(self.content_frame, bg=BG_MAIN)
        self.tab_frames["summary"] = tab

        wrapper = tk.Frame(tab, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        wrapper.pack(fill=tk.BOTH, expand=True)

        # ── Row 1: Symmetry folder ──
        row1 = tk.Frame(wrapper, bg=BG_PANEL)
        row1.pack(fill=tk.X, padx=10, pady=(10, 4))
        make_label(row1, "SYMMETRY RESULTS FOLDER").pack(side=tk.LEFT, padx=(0, 8))
        self.summary_sym_var = tk.StringVar(value=DEFAULT_SYMMETRY_DIR)
        make_entry(row1, self.summary_sym_var, width=56).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Browse", self._browse_summary_dir).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Load Tools", self._load_summary_tools, fg=FG_WARN).pack(side=tk.LEFT)

        # ── Row 2: Metadata CSV ──
        row2 = tk.Frame(wrapper, bg=BG_PANEL)
        row2.pack(fill=tk.X, padx=10, pady=(4, 4))
        make_label(row2, "TOOLS METADATA CSV").pack(side=tk.LEFT, padx=(0, 8))
        self.summary_meta_var = tk.StringVar(value=DEFAULT_TOOLS_METADATA)
        make_entry(row2, self.summary_meta_var, width=60).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row2, "Browse", self._browse_summary_meta).pack(side=tk.LEFT)

        # ── Row 3: Dual list boxes ──
        list_area = tk.Frame(wrapper, bg=BG_PANEL)
        list_area.pack(fill=tk.X, padx=10, pady=(4, 4))

        # Left column: INCLUDED
        left_col = tk.Frame(list_area, bg=BG_PANEL)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        make_label(left_col, "INCLUDED (will appear in chart)").pack(anchor="w", pady=(0, 2))
        inc_frame = tk.Frame(left_col, bg=BG_PANEL)
        inc_frame.pack(fill=tk.BOTH, expand=True)
        self.summary_include_lb = tk.Listbox(
            inc_frame, selectmode=tk.EXTENDED, height=10,
            bg=BG_ENTRY, fg=FG_MAIN, selectbackground="#0d6f2b", selectforeground="#d8ffd8",
            highlightbackground=BORDER, relief=tk.FLAT, font=("Consolas", 11),
            activestyle="none", exportselection=False,
        )
        self.summary_include_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inc_scroll = tk.Scrollbar(inc_frame, command=self.summary_include_lb.yview)
        inc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_include_lb.config(yscrollcommand=inc_scroll.set)

        # Middle column: arrow buttons
        arrow_col = tk.Frame(list_area, bg=BG_PANEL)
        arrow_col.pack(side=tk.LEFT, padx=10, anchor="center")
        tk.Frame(arrow_col, bg=BG_PANEL, height=30).pack()  # spacer
        make_button(arrow_col, ">>", self._summary_exclude_selected, width=5).pack(pady=(0, 6))
        make_button(arrow_col, "<<", self._summary_include_selected, width=5).pack(pady=(0, 6))
        make_button(arrow_col, ">> All", self._summary_exclude_all, width=5).pack(pady=(0, 6))
        make_button(arrow_col, "<< All", self._summary_include_all, width=5).pack()

        # Right column: EXCLUDED
        right_col = tk.Frame(list_area, bg=BG_PANEL)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        make_label(right_col, "EXCLUDED (skipped)").pack(anchor="w", pady=(0, 2))
        exc_frame = tk.Frame(right_col, bg=BG_PANEL)
        exc_frame.pack(fill=tk.BOTH, expand=True)
        self.summary_exclude_lb = tk.Listbox(
            exc_frame, selectmode=tk.EXTENDED, height=10,
            bg=BG_ENTRY, fg="#ff6060", selectbackground="#6f2b0d", selectforeground="#ffd8d8",
            highlightbackground=BORDER, relief=tk.FLAT, font=("Consolas", 11),
            activestyle="none", exportselection=False,
        )
        self.summary_exclude_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        exc_scroll = tk.Scrollbar(exc_frame, command=self.summary_exclude_lb.yview)
        exc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_exclude_lb.config(yscrollcommand=exc_scroll.set)

        # ── Row 4: Formats + Run ──
        row4 = tk.Frame(wrapper, bg=BG_PANEL)
        row4.pack(fill=tk.X, padx=10, pady=(6, 4))
        make_label(row4, "Output Formats").pack(side=tk.LEFT, padx=(0, 8))
        self.summary_fmt_png_var = tk.BooleanVar(value=True)
        self.summary_fmt_svg_var = tk.BooleanVar(value=False)
        self.summary_fmt_pdf_var = tk.BooleanVar(value=False)
        for text, var in [("PNG", self.summary_fmt_png_var), ("SVG", self.summary_fmt_svg_var), ("PDF", self.summary_fmt_pdf_var)]:
            tk.Checkbutton(
                row4, text=text, variable=var,
                bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
                activeforeground=FG_MAIN, font=("Consolas", 10),
            ).pack(side=tk.LEFT, padx=(0, 6))

        # ── Row 4b: Threshold controls ──
        row4b = tk.Frame(wrapper, bg=BG_PANEL)
        row4b.pack(fill=tk.X, padx=10, pady=(4, 4))
        self.summary_show_threshold_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row4b, text="Show Threshold", variable=self.summary_show_threshold_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 8))
        make_label(row4b, "Value").pack(side=tk.LEFT, padx=(0, 4))
        self.summary_threshold_var = tk.DoubleVar(value=6261.0)
        tk.Spinbox(
            row4b, from_=0, to=100000, textvariable=self.summary_threshold_var, width=10,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        row5 = tk.Frame(wrapper, bg=BG_PANEL)
        row5.pack(fill=tk.X, padx=10, pady=(4, 4))
        self.summary_run_btn = make_button(
            row5, "Generate Summary Bar Chart", self._run_summary_analysis, width=32, fg=FG_WARN,
        )
        self.summary_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.summary_count_label = tk.Label(
            row5, text="", bg=BG_PANEL, fg=FG_DIM, font=("Consolas", 10),
        )
        self.summary_count_label.pack(side=tk.LEFT)

        make_label(wrapper, "SYMMETRY SUMMARY LOG").pack(anchor="w", padx=10, pady=(6, 2))
        self.summary_log_widget = make_log(wrapper, height=10)
        self.summary_log_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._summary_log("Symmetry Summary tab ready.\n1. Set the symmetry folder and click Load Tools.\n2. Move unwanted tools to the EXCLUDED box.\n3. Click Generate.\n")

    # ── Dual-listbox helpers ──

    def _load_summary_tools(self):
        sym_root = self.summary_sym_var.get().strip()
        if not os.path.isdir(sym_root):
            messagebox.showerror("Path", f"Symmetry folder not found:\n{sym_root}")
            return

        tool_ids = []
        for entry in sorted(os.listdir(sym_root)):
            sub = os.path.join(sym_root, entry)
            if not os.path.isdir(sub):
                continue
            has_meta = any(f.endswith("_symmetry_metadata.json") for f in os.listdir(sub))
            if has_meta:
                # extract tool_id from the metadata file
                for f in os.listdir(sub):
                    if f.endswith("_symmetry_metadata.json"):
                        try:
                            with open(os.path.join(sub, f), "r", encoding="utf-8") as fp:
                                meta = json.load(fp)
                            tool_ids.append(meta.get("tool_id", entry))
                        except Exception:
                            tool_ids.append(entry)
                        break

        self.summary_include_lb.delete(0, tk.END)
        self.summary_exclude_lb.delete(0, tk.END)
        for tid in sorted(tool_ids):
            self.summary_include_lb.insert(tk.END, tid)

        self._update_summary_count()
        self._summary_log(f"Loaded {len(tool_ids)} tools from {sym_root}.\n")

    def _update_summary_count(self):
        inc = self.summary_include_lb.size()
        exc = self.summary_exclude_lb.size()
        self.summary_count_label.config(text=f"Include: {inc}   Excluded: {exc}")

    def _summary_exclude_selected(self):
        """Move selected items from Include → Exclude."""
        selected = list(self.summary_include_lb.curselection())
        items = [self.summary_include_lb.get(i) for i in selected]
        for i in reversed(selected):
            self.summary_include_lb.delete(i)
        existing = list(self.summary_exclude_lb.get(0, tk.END))
        for item in items:
            if item not in existing:
                self.summary_exclude_lb.insert(tk.END, item)
                existing.append(item)
        self._update_summary_count()

    def _summary_include_selected(self):
        """Move selected items from Exclude → Include."""
        selected = list(self.summary_exclude_lb.curselection())
        items = [self.summary_exclude_lb.get(i) for i in selected]
        for i in reversed(selected):
            self.summary_exclude_lb.delete(i)
        existing = list(self.summary_include_lb.get(0, tk.END))
        for item in items:
            if item not in existing:
                self.summary_include_lb.insert(tk.END, item)
                existing.append(item)
        self._update_summary_count()

    def _summary_exclude_all(self):
        """Move everything from Include → Exclude."""
        items = list(self.summary_include_lb.get(0, tk.END))
        self.summary_include_lb.delete(0, tk.END)
        existing = list(self.summary_exclude_lb.get(0, tk.END))
        for item in items:
            if item not in existing:
                self.summary_exclude_lb.insert(tk.END, item)
        self._update_summary_count()

    def _summary_include_all(self):
        """Move everything from Exclude → Include."""
        items = list(self.summary_exclude_lb.get(0, tk.END))
        self.summary_exclude_lb.delete(0, tk.END)
        existing = list(self.summary_include_lb.get(0, tk.END))
        for item in items:
            if item not in existing:
                self.summary_include_lb.insert(tk.END, item)
        self._update_summary_count()

    def _summary_log(self, text):
        self.summary_log_widget.insert(tk.END, text)
        self.summary_log_widget.see(tk.END)

    def _summary_log_ts(self, text):
        self.after(0, lambda: self._summary_log(text))

    def _browse_summary_dir(self):
        cur = self.summary_sym_var.get().strip() or DEFAULT_SYMMETRY_DIR
        folder = filedialog.askdirectory(initialdir=cur, title="Select symmetry results folder")
        if folder:
            self.summary_sym_var.set(folder)

    def _browse_summary_meta(self):
        cur = self.summary_meta_var.get().strip() or DEFAULT_TOOLS_METADATA
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(cur), title="Select tools_metadata.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.summary_meta_var.set(path)

    def _run_summary_analysis(self):
        if self.summary_running:
            messagebox.showwarning("Busy", "Summary analysis is already running.")
            return

        sym_root = self.summary_sym_var.get().strip()
        meta_csv = self.summary_meta_var.get().strip()

        if not os.path.isdir(sym_root):
            messagebox.showerror("Path", f"Symmetry folder not found:\n{sym_root}")
            return
        if not os.path.isfile(meta_csv):
            messagebox.showerror("Path", f"Tools metadata CSV not found:\n{meta_csv}")
            return

        include_tools = list(self.summary_include_lb.get(0, tk.END))
        if not include_tools:
            messagebox.showerror("No Tools", "The INCLUDED list is empty. Load tools first, or move some back from Excluded.")
            return

        fmts = []
        if self.summary_fmt_png_var.get():
            fmts.append("png")
        if self.summary_fmt_svg_var.get():
            fmts.append("svg")
        if self.summary_fmt_pdf_var.get():
            fmts.append("pdf")
        if not fmts:
            messagebox.showerror("Output Format", "Select at least one output format.")
            return

        cfg = OffsetAnalysisConfig(
            output_formats=tuple(fmts),
            title_font_size=max(1, int(self.offset_title_font_var.get())),
            axis_label_font_size=max(1, int(self.offset_axis_font_var.get())),
            tick_font_size=max(1, int(self.offset_tick_font_var.get())),
            legend_font_size=max(1, int(self.offset_legend_font_var.get())),
            include_top_caption=bool(self.offset_include_caption_var.get()),
        )

        self.summary_running = True
        self.summary_run_btn.config(state=tk.DISABLED)
        self._summary_log("\n" + "=" * 90 + "\n")
        self._summary_log(f"Generating summary from: {sym_root}  ({len(include_tools)} tools included)\n")
        
        threshold_value = self.summary_threshold_var.get() if self.summary_show_threshold_var.get() else None
        show_threshold = self.summary_show_threshold_var.get()
        if show_threshold:
            self._summary_log(f"Threshold: {threshold_value}\n")

        threading.Thread(
            target=self._summary_analysis_worker,
            args=(sym_root, meta_csv, cfg, include_tools, threshold_value, show_threshold),
            daemon=True,
        ).start()

    def _summary_analysis_worker(self, sym_root, meta_csv, cfg, include_tools, threshold_value=None, show_threshold=False):
        try:
            result = run_symmetry_summary(
                sym_root, meta_csv, cfg,
                log_fn=self._summary_log_ts,
                include_tools=include_tools,
                threshold_value=threshold_value,
                show_threshold=show_threshold,
            )
            self._summary_log_ts(
                f"\nDone! {result['tool_count']} tools in chart.\n"
                f"Plots: {', '.join(result['plot_paths'])}\n"
                f"CSV: {result['csv_path']}\n"
            )
        except Exception as exc:
            self._summary_log_ts(f"\nERROR: {exc}\n")
        finally:
            self.summary_running = False
            self.after(0, lambda: self.summary_run_btn.config(state=tk.NORMAL))

    # ================================================================
    # TAB 5: CUSTOM SUMMARY GRAPH
    # ================================================================
    def _build_custom_graph_tab(self):
        tab = tk.Frame(self.content_frame, bg=BG_MAIN)
        self.tab_frames["custom"] = tab

        wrapper = tk.Frame(tab, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        wrapper.pack(fill=tk.BOTH, expand=True)

        # ── Row 1: Symmetry folder & Metadata CSV ──
        row1 = tk.Frame(wrapper, bg=BG_PANEL)
        row1.pack(fill=tk.X, padx=10, pady=(10, 4))
        make_label(row1, "SYMMETRY RESULTS FOLDER").pack(side=tk.LEFT, padx=(0, 8))
        make_entry(row1, self.custom_graph_sym_var, width=48).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Browse", self._browse_custom_graph_dir).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row1, "Load Tools", self._load_custom_graph_tools, fg=FG_WARN).pack(side=tk.LEFT)

        row2 = tk.Frame(wrapper, bg=BG_PANEL)
        row2.pack(fill=tk.X, padx=10, pady=(4, 4))
        make_label(row2, "TOOLS METADATA CSV").pack(side=tk.LEFT, padx=(0, 8))
        make_entry(row2, self.custom_graph_meta_var, width=56).pack(side=tk.LEFT, padx=(0, 6))
        make_button(row2, "Browse", self._browse_custom_graph_meta).pack(side=tk.LEFT)

        # ── Row 3: Number of labels ──
        row3 = tk.Frame(wrapper, bg=BG_PANEL)
        row3.pack(fill=tk.X, padx=10, pady=(4, 4))
        make_label(row3, "Number of Labels").pack(side=tk.LEFT, padx=(0, 4))
        self.custom_graph_labels_spin = tk.Spinbox(
            row3, from_=1, to=10, textvariable=self.custom_graph_num_labels_var, width=4,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10), command=self._rebuild_custom_graph_labels,
        )
        self.custom_graph_labels_spin.pack(side=tk.LEFT, padx=(0, 12))
        make_button(row3, "Update Labels", self._rebuild_custom_graph_labels).pack(side=tk.LEFT)

        # ── Row 3b: Chart title ──
        row3b = tk.Frame(wrapper, bg=BG_PANEL)
        row3b.pack(fill=tk.X, padx=10, pady=(4, 4))
        tk.Checkbutton(
            row3b, text="Show Title", variable=self.custom_graph_show_title_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 12))
        make_label(row3b, "Title").pack(side=tk.LEFT, padx=(0, 4))
        make_entry(row3b, self.custom_graph_title_var, width=50).pack(side=tk.LEFT)

        # ── Row 4: Labels configuration container ──
        make_label(wrapper, "LABEL CONFIGURATION").pack(anchor="w", padx=10, pady=(6, 2))
        self.custom_graph_labels_frame = tk.Frame(wrapper, bg=BG_PANEL)
        self.custom_graph_labels_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        # Initialize with default number of labels
        self._rebuild_custom_graph_labels()

        # ── Row 5: Output formats ──
        row_fmt = tk.Frame(wrapper, bg=BG_PANEL)
        row_fmt.pack(fill=tk.X, padx=10, pady=(4, 4))
        make_label(row_fmt, "Output Formats").pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(
            row_fmt, text="PNG", variable=self.custom_graph_fmt_png_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(
            row_fmt, text="SVG", variable=self.custom_graph_fmt_svg_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(
            row_fmt, text="PDF", variable=self.custom_graph_fmt_pdf_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        # ── Row 5b: Threshold controls ──
        row_thresh = tk.Frame(wrapper, bg=BG_PANEL)
        row_thresh.pack(fill=tk.X, padx=10, pady=(4, 4))
        tk.Checkbutton(
            row_thresh, text="Show Threshold", variable=self.custom_graph_show_threshold_var,
            bg=BG_PANEL, fg=FG_MAIN, selectcolor=BG_ENTRY, activebackground=BG_PANEL,
            activeforeground=FG_MAIN, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 8))
        make_label(row_thresh, "Value").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row_thresh, from_=0, to=100000, textvariable=self.custom_graph_threshold_var, width=10,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        # ── Row 6: Font sizes ──
        row_fonts = tk.Frame(wrapper, bg=BG_PANEL)
        row_fonts.pack(fill=tk.X, padx=10, pady=(4, 4))
        make_label(row_fonts, "Title Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row_fonts, from_=8, to=48, textvariable=self.custom_graph_title_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))
        make_label(row_fonts, "Axis Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row_fonts, from_=8, to=48, textvariable=self.custom_graph_axis_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))
        make_label(row_fonts, "Tick Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row_fonts, from_=6, to=36, textvariable=self.custom_graph_tick_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=(0, 10))
        make_label(row_fonts, "Legend Font").pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row_fonts, from_=6, to=48, textvariable=self.custom_graph_legend_font_var, width=5,
            bg=BG_ENTRY, fg=FG_MAIN, insertbackground=FG_MAIN, buttonbackground="#0b2a0f",
            relief=tk.FLAT, font=("Consolas", 10),
        ).pack(side=tk.LEFT)

        # ── Row 7: Run button ──
        row_run = tk.Frame(wrapper, bg=BG_PANEL)
        row_run.pack(fill=tk.X, padx=10, pady=(6, 4))
        self.custom_graph_run_btn = make_button(
            row_run, "Generate Custom Summary", self._run_custom_graph_analysis, width=32, fg=FG_WARN,
        )
        self.custom_graph_run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.custom_graph_stop_btn = make_button(row_run, "Stop", self._stop_custom_graph_analysis, width=10)
        self.custom_graph_stop_btn.pack(side=tk.LEFT)

        # ── Row 8: Log ──
        make_label(wrapper, "CUSTOM GRAPH LOG").pack(anchor="w", padx=10, pady=(6, 2))
        self.custom_graph_log_widget = make_log(wrapper, height=10)
        self.custom_graph_log_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._custom_graph_log("Custom Summary tab ready.\n1. Load symmetry folder and tools.\n2. Configure labels, colors, and tool assignments.\n3. Click Generate.\n")

    def _rebuild_custom_graph_labels(self):
        """Rebuild labels frame with drag-drop tool assignment."""
        # Clear existing
        for child in self.custom_graph_labels_frame.winfo_children():
            child.destroy()
        self.custom_graph_label_vars = []
        self.custom_graph_tool_assignments = {}
        
        num_labels = max(1, int(self.custom_graph_num_labels_var.get()))

        # Two-column layout: Available Tools (left) | Label Configs (right)
        container = tk.Frame(self.custom_graph_labels_frame, bg=BG_PANEL)
        container.pack(fill=tk.BOTH, expand=True)

        # ──── LEFT COLUMN: Available Tools ────
        left_frame = tk.Frame(container, bg=BG_PANEL)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        make_label(left_frame, "Available Tools (select & drag to label)").pack(anchor="w", padx=2, pady=(0, 4))
        self.custom_graph_available_tools_listbox = tk.Listbox(
            left_frame, selectmode=tk.EXTENDED, height=20, width=30,
            bg=BG_ENTRY, fg=FG_MAIN, selectbackground="#0d6f2b", selectforeground="#d8ffd8",
            highlightbackground=BORDER, relief=tk.FLAT, font=("Consolas", 9),
            activestyle="none", exportselection=False,
        )
        self.custom_graph_available_tools_listbox.pack(fill=tk.BOTH, expand=True)

        # ──── RIGHT COLUMN: Label drop targets ────
        right_frame = tk.Frame(container, bg=BG_PANEL)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Scrollable frame for label configs
        canvas = tk.Canvas(right_frame, bg=BG_PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(right_frame, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG_PANEL)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for i in range(num_labels):
            self._create_label_drop_target(scrollable_frame, i)
            self.custom_graph_tool_assignments[i] = []

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_label_drop_target(self, parent, label_idx):
        """Create a label config with drop target for tools."""
        row = tk.Frame(parent, bg=BG_PANEL, relief=tk.FLAT)
        row.pack(fill=tk.X, padx=4, pady=4)

        # Header frame (label name + color)
        header = tk.Frame(row, bg=BG_PANEL)
        header.pack(fill=tk.X, padx=4, pady=(4, 2))

        make_label(header, f"Label {label_idx + 1}").pack(side=tk.LEFT, padx=(0, 6))
        
        # Default label names
        default_names = ["New", "Used", "Used (Segmentation Problem)", "Fractured"]
        if label_idx < len(default_names):
            default_name = default_names[label_idx]
        else:
            default_name = f"Category {label_idx + 1}"
        
        name_var = tk.StringVar(value=default_name)
        name_entry = make_entry(header, name_var, width=25)
        name_entry.pack(side=tk.LEFT, padx=(0, 6))

        # Color picker - Default colors: dark green, light green, yellow, red
        default_colors = ["#2d5a2d", "#90ee90", "#ffdd00", "#ff0000"]
        default_color = default_colors[label_idx] if label_idx < len(default_colors) else "#2ca02c"
        color_var = tk.StringVar(value=default_color)
        color_display = tk.Label(header, text="   ", bg=color_var.get(), width=5)
        color_display.pack(side=tk.LEFT, padx=(0, 6))

        def pick_color(cv=color_var, cd=color_display):
            from tkinter import colorchooser
            color = colorchooser.askcolor(color=cv.get())
            if color[1]:
                cv.set(color[1])
                cd.config(bg=color[1])

        make_button(header, "Color", pick_color, width=10).pack(side=tk.LEFT)

        # Drop target frame (for assigned tools)
        drop_frame = tk.Frame(row, bg=BG_ENTRY, relief=tk.SUNKEN, bd=1)
        drop_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4), ipady=4)
        
        # Label to show assigned tools
        tools_display = tk.Frame(drop_frame, bg=BG_ENTRY)
        tools_display.pack(fill=tk.BOTH, expand=True)
        
        # Store references
        self.custom_graph_label_vars.append((name_var, color_var, tools_display, label_idx))

        # Setup drag-drop for this drop target
        self._setup_drop_target(tools_display, label_idx)

    def _setup_drop_target(self, drop_frame, label_idx):
        """Setup drag-drop event handlers for a drop target frame."""
        drop_frame.bind("<Button-1>", lambda e: self._on_tool_drag_start(e, label_idx), add="+")
        drop_frame.bind("<B1-Motion>", self._on_tool_drag_motion, add="+")
        drop_frame.bind("<ButtonRelease-1>", lambda e: self._on_tool_drop(e, label_idx), add="+")
        drop_frame.bind("<Button-3>", lambda e: self._on_tool_right_click(e, label_idx), add="+")

    def _on_tool_drag_start(self, event, label_idx):
        """Start dragging tools from available tools listbox (supports multi-select)."""
        selections = self.custom_graph_available_tools_listbox.curselection()
        if selections:
            self.custom_graph_dragging_tools = [self.custom_graph_available_tools_listbox.get(idx) for idx in selections]

    def _on_tool_drag_motion(self, event):
        """Motion during drag (visual feedback could go here)."""
        pass

    def _on_tool_drop(self, event, label_idx):
        """Drop all dragged tools into label's drop target."""
        if self.custom_graph_dragging_tools:
            for tool_id in self.custom_graph_dragging_tools:
                if tool_id not in self.custom_graph_tool_assignments[label_idx]:
                    self.custom_graph_tool_assignments[label_idx].append(tool_id)
            self._update_label_display(label_idx)
            self.custom_graph_dragging_tools = []

    def _on_tool_right_click(self, event, label_idx):
        """Right-click to remove tool from label."""
        # Find which tool widget was clicked
        widget = event.widget
        if isinstance(widget, tk.Label) and hasattr(widget, "_tool_id"):
            tool_id = widget._tool_id
            if tool_id in self.custom_graph_tool_assignments[label_idx]:
                self.custom_graph_tool_assignments[label_idx].remove(tool_id)
                self._update_label_display(label_idx)

    def _update_label_display(self, label_idx):
        """Update the display of tools for a label."""
        # Find the display frame for this label
        for name_var, color_var, tools_display, idx in self.custom_graph_label_vars:
            if idx == label_idx:
                # Clear existing tool widgets
                for child in tools_display.winfo_children():
                    child.destroy()
                
                # Add tool label widgets
                tools = self.custom_graph_tool_assignments.get(label_idx, [])
                for tool_id in tools:
                    tool_label = tk.Label(
                        tools_display, text=tool_id, bg="#0d6f2b", fg=FG_MAIN,
                        padx=6, pady=2, font=("Consolas", 9), relief=tk.RAISED
                    )
                    tool_label._tool_id = tool_id
                    tool_label.pack(side=tk.LEFT, padx=2, pady=2)
                    tool_label.bind("<Button-3>", lambda e: self._on_tool_right_click(e, label_idx))
                break

    def _load_custom_graph_tools(self):
        """Load available tools from symmetry folder into central listbox."""
        sym_root = self.custom_graph_sym_var.get().strip()
        if not os.path.isdir(sym_root):
            messagebox.showerror("Path", f"Symmetry folder not found:\n{sym_root}")
            return

        tool_ids = []
        for entry in sorted(os.listdir(sym_root)):
            sub = os.path.join(sym_root, entry)
            if not os.path.isdir(sub):
                continue
            has_meta = any(f.endswith("_symmetry_metadata.json") for f in os.listdir(sub))
            if has_meta:
                for f in os.listdir(sub):
                    if f.endswith("_symmetry_metadata.json"):
                        try:
                            with open(os.path.join(sub, f), "r", encoding="utf-8") as fp:
                                meta = json.load(fp)
                            tool_ids.append(meta.get("tool_id", entry))
                        except Exception:
                            tool_ids.append(entry)
                        break

        tool_ids = sorted(tool_ids)

        # Populate the central available tools listbox
        if self.custom_graph_available_tools_listbox:
            self.custom_graph_available_tools_listbox.delete(0, tk.END)
            for tid in tool_ids:
                self.custom_graph_available_tools_listbox.insert(tk.END, tid)

        self._custom_graph_log(f"Loaded {len(tool_ids)} tools.\n")

    def _browse_custom_graph_dir(self):
        cur = self.custom_graph_sym_var.get().strip() or DEFAULT_SYMMETRY_DIR
        folder = filedialog.askdirectory(initialdir=cur, title="Select symmetry results folder")
        if folder:
            self.custom_graph_sym_var.set(folder)

    def _browse_custom_graph_meta(self):
        cur = self.custom_graph_meta_var.get().strip() or DEFAULT_TOOLS_METADATA
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(cur), title="Select tools_metadata.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.custom_graph_meta_var.set(path)

    def _custom_graph_log(self, text):
        self.custom_graph_log_widget.insert(tk.END, text)
        self.custom_graph_log_widget.see(tk.END)

    def _custom_graph_log_ts(self, text):
        self.after(0, lambda: self._custom_graph_log(text))

    def _run_custom_graph_analysis(self):
        if self.custom_graph_running:
            messagebox.showwarning("Busy", "Custom graph generation is already running.")
            return

        sym_root = self.custom_graph_sym_var.get().strip()
        meta_csv = self.custom_graph_meta_var.get().strip()

        if not os.path.isdir(sym_root):
            messagebox.showerror("Path", f"Symmetry folder not found:\n{sym_root}")
            return
        if not os.path.isfile(meta_csv):
            messagebox.showerror("Path", f"Tools metadata CSV not found:\n{meta_csv}")
            return

        # Build label configuration from UI (using tool_assignments dict)
        labels_config = []
        for name_var, color_var, tools_display, idx in self.custom_graph_label_vars:
            label_name = name_var.get().strip() or f"Category {idx + 1}"
            label_color = color_var.get().strip()
            assigned_tools = self.custom_graph_tool_assignments.get(idx, [])

            if assigned_tools:
                labels_config.append({
                    "name": label_name,
                    "color": label_color,
                    "tools": assigned_tools,
                })

        if not labels_config:
            messagebox.showerror("Configuration", "Please assign tools to at least one label.")
            return

        fmts = []
        if self.custom_graph_fmt_png_var.get():
            fmts.append("png")
        if self.custom_graph_fmt_svg_var.get():
            fmts.append("svg")
        if self.custom_graph_fmt_pdf_var.get():
            fmts.append("pdf")
        if not fmts:
            messagebox.showerror("Output Format", "Select at least one output format.")
            return

        cfg = OffsetAnalysisConfig(
            output_formats=tuple(fmts),
            title_font_size=max(1, int(self.custom_graph_title_font_var.get())),
            axis_label_font_size=max(1, int(self.custom_graph_axis_font_var.get())),
            tick_font_size=max(1, int(self.custom_graph_tick_font_var.get())),
            legend_font_size=max(1, int(self.custom_graph_legend_font_var.get())),
        )

        threshold_value = self.custom_graph_threshold_var.get() if self.custom_graph_show_threshold_var.get() else None
        show_threshold = self.custom_graph_show_threshold_var.get()
        
        custom_title = self.custom_graph_title_var.get().strip() if self.custom_graph_show_title_var.get() else None
        show_title = self.custom_graph_show_title_var.get()

        self.custom_graph_running = True
        self.custom_graph_run_btn.config(state=tk.DISABLED)
        self._custom_graph_log("\n" + "=" * 90 + "\n")
        self._custom_graph_log(f"Generating custom summary with {len(labels_config)} label(s)...\n")
        if show_threshold:
            self._custom_graph_log(f"Threshold: {threshold_value}\n")
        if show_title and custom_title:
            self._custom_graph_log(f"Title: {custom_title}\n")

        threading.Thread(
            target=self._custom_graph_worker,
            args=(sym_root, meta_csv, cfg, labels_config, threshold_value, show_threshold, custom_title, show_title),
            daemon=True,
        ).start()

    def _custom_graph_worker(self, sym_root, meta_csv, cfg, labels_config, threshold_value=None, show_threshold=False, custom_title=None, show_title=False):
        try:
            result = run_custom_summary_graph(
                sym_root, meta_csv, cfg, labels_config,
                threshold_value=threshold_value,
                show_threshold=show_threshold,
                custom_title=custom_title,
                show_title=show_title,
                log_fn=self._custom_graph_log_ts,
            )
            self._custom_graph_log_ts(
                f"\nDone! {result['tool_count']} tools in chart.\n"
                f"Plots: {', '.join(result['plot_paths'])}\n"
                f"CSV: {result['csv_path']}\n"
            )
        except Exception as exc:
            self._custom_graph_log_ts(f"\nERROR: {exc}\n")
        finally:
            self.custom_graph_running = False
            self.after(0, lambda: self.custom_graph_run_btn.config(state=tk.NORMAL))

    def _stop_custom_graph_analysis(self):
        if not self.custom_graph_running:
            self._custom_graph_log("No active custom graph generation.\n")
            return
        # Note: Threading doesn't support direct interrupt; user can just close or restart
        self._custom_graph_log("Stop requested.\n")

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

    def _load_roi_context(self):
        sel = self.roi_tool_combo.curselection()
        if not sel:
            messagebox.showwarning("Selection", "Select a tool folder first.")
            return None

        tool_folder_name = self.roi_tool_combo.get(sel[0])
        tilted_root = self.roi_tilted_var.get().strip()
        tool_dir = os.path.join(tilted_root, tool_folder_name)
        info_dir = os.path.join(tool_dir, "information")

        if not os.path.isdir(tool_dir):
            self._roi_log(f"Tool directory missing: {tool_dir}\n")
            return None

        master_files = [
            f for f in os.listdir(info_dir)
            if "MASTER_MASK" in f.upper() and f.lower().endswith(".png")
        ] if os.path.isdir(info_dir) else []
        if not master_files:
            self._roi_log(f"No master mask PNG found in {info_dir}\n")
            return None

        master_path = os.path.join(info_dir, master_files[0])
        master = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
        if master is None:
            self._roi_log(f"Cannot read master mask: {master_path}\n")
            return None
        _, master_bin = cv2.threshold(master, 127, 255, cv2.THRESH_BINARY)

        meta_files = [f for f in os.listdir(info_dir) if f.endswith("_tilt_metadata.json")]
        if not meta_files:
            self._roi_log(
                "No *_tilt_metadata.json found. Run Tab 1 first so ROI geometry is saved.\n"
            )
            return None

        meta_path = os.path.join(info_dir, meta_files[0])
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)

        frame_files = sorted(
            f for f in os.listdir(tool_dir)
            if f.lower().endswith(".png") and os.path.isfile(os.path.join(tool_dir, f))
        )
        if not frame_files:
            self._roi_log("No tilted PNG frames found.\n")
            return None

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
            return None
        _, frame_bin = cv2.threshold(frame_img, 127, 255, cv2.THRESH_BINARY)

        return {
            "tool_folder_name": tool_folder_name,
            "tool_dir": tool_dir,
            "info_dir": info_dir,
            "meta_path": meta_path,
            "meta": meta,
            "master_bin": master_bin,
            "frame_bin": frame_bin,
            "frame_files": frame_files,
            "frame_idx": frame_idx,
            "frame_name": frame_files[frame_idx],
        }

    @staticmethod
    def _clamp_geo(geo, h, w):
        geo["roi_top"] = max(0, min(int(geo["roi_top"]), h - 1))
        geo["roi_bottom"] = max(geo["roi_top"] + 1, min(int(geo["roi_bottom"]), h))
        geo["roi_left"] = max(0, min(int(geo["roi_left"]), w - 1))
        geo["roi_right"] = max(geo["roi_left"] + 1, min(int(geo["roi_right"]), w))
        geo["center_x"] = max(geo["roi_left"], min(int(geo["center_x"]), geo["roi_right"] - 1))
        geo["width"] = geo["roi_right"] - geo["roi_left"]
        geo["height"] = geo["roi_bottom"] - geo["roi_top"]
        return geo

    def _roi_method2_geo(self, frame_bin):
        ys = np.where(frame_bin.any(axis=1))[0]
        if len(ys) == 0:
            self._roi_log("Selected frame is empty after thresholding.\n")
            return None

        bottom_y = int(ys[-1])
        cols_full = np.where(frame_bin.any(axis=0))[0]
        if len(cols_full) < 2:
            self._roi_log("Selected frame is too narrow for ROI2.\n")
            return None

        # Iterative frame-dependent ROI: width -> height -> ROI rows -> width.
        width = int(cols_full[-1] - cols_full[0])
        roi_height = max(1, int(round(0.45 * width)))

        for _ in range(3):
            roi_top = max(0, bottom_y - roi_height)
            roi_slice = frame_bin[roi_top:bottom_y, :]
            if roi_slice.size == 0:
                break
            cols_roi = np.where(roi_slice.any(axis=0))[0]
            if len(cols_roi) < 2:
                break
            width_new = int(cols_roi[-1] - cols_roi[0])
            roi_height_new = max(1, int(round(0.45 * width_new)))
            if width_new == width and roi_height_new == roi_height:
                width = width_new
                roi_height = roi_height_new
                break
            width = width_new
            roi_height = roi_height_new

        roi_top = max(0, bottom_y - roi_height)
        roi_slice = frame_bin[roi_top:bottom_y, :]
        cols_roi = np.where(roi_slice.any(axis=0))[0]
        if len(cols_roi) < 2:
            cols_roi = cols_full

        roi_left = int(cols_roi[0])
        roi_right = int(cols_roi[-1])
        center_x = int(round((roi_left + roi_right) / 2.0))

        geo = {
            "method": "ROI2",
            "roi_top": roi_top,
            "roi_bottom": bottom_y,
            "roi_left": roi_left,
            "roi_right": roi_right,
            "center_x": center_x,
            "mask_width": int(roi_right - roi_left),
            "roi_height_formula": "0.45 * frame-dependent ROI width",
        }
        h, w = frame_bin.shape
        return self._clamp_geo(geo, h, w)

    @staticmethod
    def _render_overlay(frame_bin, geo):
        vis = cv2.cvtColor(frame_bin, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0
        alpha = 0.35

        blue_mask = np.zeros_like(vis)
        blue_mask[geo["roi_top"]:geo["roi_bottom"], geo["roi_left"]:geo["center_x"]] = [0.0, 0.0, 1.0]
        red_mask = np.zeros_like(vis)
        red_mask[geo["roi_top"]:geo["roi_bottom"], geo["center_x"]:geo["roi_right"]] = [1.0, 0.0, 0.0]

        bm = blue_mask.sum(axis=2) > 0
        rm = red_mask.sum(axis=2) > 0
        vis[bm] = vis[bm] * (1 - alpha) + blue_mask[bm] * alpha
        vis[rm] = vis[rm] * (1 - alpha) + red_mask[rm] * alpha
        return vis

    @staticmethod
    def _draw_roi_guides(ax, geo):
        from matplotlib.patches import Rectangle as MplRect

        rect = MplRect(
            (geo["roi_left"], geo["roi_top"]),
            geo["roi_right"] - geo["roi_left"],
            geo["roi_bottom"] - geo["roi_top"],
            linewidth=2.2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(
            [geo["center_x"], geo["center_x"]],
            [geo["roi_top"], geo["roi_bottom"]],
            color="yellow",
            linewidth=2.2,
        )

    def _render_roi_preview_in_frame(self, frame_bin, geo):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self.roi_preview_canvas is not None:
            self.roi_preview_canvas.get_tk_widget().destroy()
            self.roi_preview_canvas = None
            self.roi_preview_fig = None

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.subplots(1, 1)
        vis = self._render_overlay(frame_bin, geo)
        ax.imshow(vis, origin="upper")
        self._draw_roi_guides(ax, geo)
        ax.set_axis_off()

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        canvas = FigureCanvasTkAgg(fig, master=self.roi_preview_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw_idle()

        self.roi_preview_canvas = canvas
        self.roi_preview_fig = fig

    def _on_roi_preview_resize(self, event):
        if self.roi_preview_canvas is None or self.roi_preview_fig is None:
            return
        if event.width < 80 or event.height < 80:
            return

        dpi = self.roi_preview_fig.get_dpi()
        self.roi_preview_fig.set_size_inches(event.width / dpi, event.height / dpi, forward=True)
        self.roi_preview_canvas.draw_idle()

    def _save_roi_image_no_text(self, frame_bin, geo, out_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis = self._render_overlay(frame_bin, geo)
        h, w = frame_bin.shape
        fig, ax = plt.subplots(figsize=(max(2, w / 140.0), max(2, h / 140.0)), dpi=140)
        ax.imshow(vis, origin="upper")
        self._draw_roi_guides(ax, geo)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    def _save_roi_image_with_legend(self, frame_bin, geo, out_path, caption):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MplRect
        from matplotlib.lines import Line2D

        vis = self._render_overlay(frame_bin, geo)
        fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
        ax.imshow(vis, origin="upper")

        rect = MplRect(
            (geo["roi_left"], geo["roi_top"]),
            geo["roi_right"] - geo["roi_left"],
            geo["roi_bottom"] - geo["roi_top"],
            linewidth=2.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(
            [geo["center_x"], geo["center_x"]],
            [geo["roi_top"], geo["roi_bottom"]],
            color="yellow",
            linewidth=2.5,
        )

        legend_handles = [
            MplRect((0, 0), 1, 1, facecolor="blue", alpha=0.5, label="Left ROI"),
            MplRect((0, 0), 1, 1, facecolor="red", alpha=0.5, label="Right ROI"),
            Line2D([0], [0], color="yellow", linewidth=2.5, label="Centerline in ROI"),
            MplRect((0, 0), 1, 1, edgecolor="lime", facecolor="none", linewidth=2, label="ROI Box"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=11)
        if caption:
            ax.set_title(caption, fontsize=18)
        ax.set_axis_off()
        fig.tight_layout(pad=0.15)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _get_move_step(self):
        try:
            return max(1, int(self.roi_step_var.get()))
        except Exception:
            return 1

    def _update_adjust_buttons_state(self):
        target = self.roi_edit_target_var.get()
        horizontal = target in {"left", "right", "centerline"}
        vertical = target in {"top", "bottom"}
        self.btn_move_left.config(state=tk.NORMAL if horizontal else tk.DISABLED)
        self.btn_move_right.config(state=tk.NORMAL if horizontal else tk.DISABLED)
        self.btn_move_up.config(state=tk.NORMAL if vertical else tk.DISABLED)
        self.btn_move_down.config(state=tk.NORMAL if vertical else tk.DISABLED)

    def _move_selected_edge(self, dx, dy):
        if self.roi_current_context is None or self.roi_current_geo is None:
            self._roi_log("No ROI loaded yet. Click Preview ROI first.\n")
            return

        target = self.roi_edit_target_var.get()
        if target in {"left", "right", "centerline"} and dy != 0:
            return
        if target in {"top", "bottom"} and dx != 0:
            return

        geo = dict(self.roi_current_geo)
        if target == "left":
            geo["roi_left"] += dx
        elif target == "right":
            geo["roi_right"] += dx
        elif target == "centerline":
            geo["center_x"] += dx
        elif target == "top":
            geo["roi_top"] += dy
        elif target == "bottom":
            geo["roi_bottom"] += dy

        h, w = self.roi_current_context["frame_bin"].shape
        geo = self._clamp_geo(geo, h, w)
        self.roi_current_geo = geo
        self._render_roi_preview_in_frame(self.roi_current_context["frame_bin"], self.roi_current_geo)
        self._roi_log(
            f"Adjusted {target}: left={geo['roi_left']}, right={geo['roi_right']}, "
            f"top={geo['roi_top']}, bottom={geo['roi_bottom']}, center={geo['center_x']}\n"
        )

    def _persist_current_geo_to_metadata(self):
        if self.roi_current_context is None or self.roi_current_geo is None:
            return False

        ctx = self.roi_current_context
        meta = dict(ctx["meta"])
        geo = self.roi_current_geo

        meta["master_mask_width_px"] = int(geo["width"])
        meta["roi_height_px"] = int(geo["height"])
        meta["centerline_column_px"] = int(geo["center_x"])
        meta["roi_top_px"] = int(geo["roi_top"])
        meta["roi_bottom_px"] = int(geo["roi_bottom"])
        meta["roi_left_px"] = int(geo["roi_left"])
        meta["roi_right_px"] = int(geo["roi_right"])
        meta["roi_method"] = "ROI2_frame_dependent_manual"
        meta["roi_height_formula"] = "manual_adjusted_from_method2"
        meta["centerline_definition"] = "manual_adjusted_centerline_column"

        with open(ctx["meta_path"], "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

        self.roi_current_context["meta"] = meta
        return True

    def _preview_roi(self):
        ctx = self._load_roi_context()
        if ctx is None:
            return

        geo = self._roi_method2_geo(ctx["frame_bin"])
        if geo is None:
            return

        self.roi_current_context = ctx
        self.roi_current_geo = geo
        self._render_roi_preview_in_frame(ctx["frame_bin"], geo)
        self._roi_log(
            f"Preview loaded for {ctx['tool_folder_name']} | frame {ctx['frame_name']} (index {ctx['frame_idx']})\n"
            f"  width={geo['width']}px, height={geo['height']}px, center_x={geo['center_x']}\n"
        )

    def _save_roi_and_metadata(self):
        if self.roi_current_context is None or self.roi_current_geo is None:
            self._roi_log("No ROI loaded yet. Click Preview ROI first.\n")
            return

        ctx = self.roi_current_context
        geo = self.roi_current_geo
        tool_folder_name = ctx["tool_folder_name"]
        info_dir = ctx["info_dir"]
        frame_name = ctx["frame_name"]
        meta = ctx["meta"]

        match = re.search(r"(tool\d+)", tool_folder_name, re.IGNORECASE)
        tid = match.group(1) if match else tool_folder_name
        frame_stem = os.path.splitext(frame_name)[0]

        caption_raw = self.roi_caption_var.get().strip()
        if caption_raw == "-":
            caption = None
        elif caption_raw:
            caption = caption_raw
        else:
            tool_type = meta.get("tool_type", "")
            tid_num = re.sub(r"[^0-9]", "", tid)
            caption = f"{(tool_type or 'Tool').capitalize()} Tool ({tid_num})"

        roi2_dir = os.path.join(info_dir, "ROI2")
        os.makedirs(roi2_dir, exist_ok=True)
        out2 = os.path.join(roi2_dir, f"{tid}_{frame_stem}_roi2.png")

        main_out = os.path.join(info_dir, f"{tid}_roi_visualization.png")
        self._save_roi_image_with_legend(ctx["frame_bin"], geo, out2, caption)
        self._save_roi_image_with_legend(ctx["frame_bin"], geo, main_out, caption)

        # Also save ROI visualization to the symmetry folder.
        base_data = self.base_data_var.get().strip()
        sym_dir = os.path.join(base_data, "symmetry", tid.lower())
        os.makedirs(sym_dir, exist_ok=True)
        sym_out = os.path.join(sym_dir, f"{tid.lower()}_roi_visualization.png")
        self._save_roi_image_with_legend(ctx["frame_bin"], geo, sym_out, caption)

        saved_meta = self._persist_current_geo_to_metadata()
        if saved_meta:
            self._roi_log(
                f"Saved ROI image: {out2}\n"
                f"Saved ROI image: {main_out}\n"
                f"Saved ROI image: {sym_out}\n"
                f"Updated metadata: {ctx['meta_path']}\n"
                f"  left={geo['roi_left']}, right={geo['roi_right']}, "
                f"top={geo['roi_top']}, bottom={geo['roi_bottom']}, center={geo['center_x']}\n"
            )

if __name__ == "__main__":
    # Add script dir to path so ROI tab can import processing functions
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    app = MatrixPerspectiveGUI()
    app.mainloop()
