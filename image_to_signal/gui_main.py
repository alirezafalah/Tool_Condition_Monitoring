"""
Sophisticated GUI for Image-to-Signal Processing Pipeline
"""
import sys
import os
import random
import warnings
import numpy as np
import cv2
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QGroupBox, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QComboBox, QProgressBar, QTextEdit, QTabWidget,
                            QFileDialog, QFrame, QScrollArea, QDialog, QListWidget, QListWidgetItem,
                            QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QImage, QPixmap, QShortcut, QKeySequence

# Suppress matplotlib threading warnings
warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside of the main thread')

# Import processing modules
from . import step1_blur_and_rename
from . import step2_generate_masks
from . import step3_analyze_and_plot
from . import step4_process_and_plot
from . import dashboard_generator
from .utils.filters import (create_multichannel_mask, fill_holes, morph_closing,
                            keep_largest_contour, background_subtraction_absdiff,
                            background_subtraction_lab)


class ProcessingThread(QThread):
    """Thread for running processing steps without blocking UI."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, step_func, config):
        super().__init__()
        self.step_func = step_func
        self.config = config
    
    def run(self):
        try:
            # Run the step - output goes to terminal
            self.step_func(self.config)
            self.finished.emit("Step completed successfully!")
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class MaskPreviewDialog(QDialog):
    """Modeless dialog for previewing step 2 mask generation on blurred frames."""

    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    random_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Mask Preview")
        self.resize(1200, 760)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.frame_label = QLabel("Frame: -")
        self.frame_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout.addWidget(self.frame_label)

        self.status_label = QLabel("Preview uses the current blurred frame and refreshes while this window is open.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(14)

        original_layout = QVBoxLayout()
        original_title = QLabel("Blurred Frame")
        original_title.setStyleSheet("font-weight: bold;")
        original_layout.addWidget(original_title)
        self.original_image_label = QLabel("No preview loaded")
        self._configure_image_label(self.original_image_label)
        original_layout.addWidget(self.original_image_label, stretch=1)
        panels_row.addLayout(original_layout, stretch=1)

        mask_layout = QVBoxLayout()
        mask_title = QLabel("Final Mask")
        mask_title.setStyleSheet("font-weight: bold;")
        mask_layout.addWidget(mask_title)
        self.mask_image_label = QLabel("No preview loaded")
        self._configure_image_label(self.mask_image_label)
        mask_layout.addWidget(self.mask_image_label, stretch=1)
        panels_row.addLayout(mask_layout, stretch=1)

        layout.addLayout(panels_row, stretch=1)

        controls_row = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.previous_requested.emit)
        controls_row.addWidget(self.prev_btn)

        self.random_btn = QPushButton("Random")
        self.random_btn.clicked.connect(self.random_requested.emit)
        controls_row.addWidget(self.random_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_requested.emit)
        controls_row.addWidget(self.next_btn)

        controls_row.addStretch()

        help_label = QLabel("Keyboard: Left/Right arrows to navigate, R for random frame")
        help_label.setStyleSheet("color: #888;")
        controls_row.addWidget(help_label)
        layout.addLayout(controls_row)

        QShortcut(QKeySequence("Left"), self, activated=self.previous_requested.emit)
        QShortcut(QKeySequence("Right"), self, activated=self.next_requested.emit)
        QShortcut(QKeySequence("R"), self, activated=self.random_requested.emit)

    def _configure_image_label(self, label):
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(480, 520)
        label.setStyleSheet("background: #111; border: 1px solid #444; color: #888;")
        label.setWordWrap(True)

    def _set_label_pixmap(self, label, pixmap, empty_text):
        if pixmap is None:
            label.clear()
            label.setText(empty_text)
            return

        target_size = label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            scaled = pixmap
        else:
            scaled = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        label.setText("")
        label.setPixmap(scaled)

    def set_original_preview(self, pixmap, empty_text="No frame loaded"):
        self._set_label_pixmap(self.original_image_label, pixmap, empty_text)

    def set_mask_preview(self, pixmap, empty_text="No mask preview available"):
        self._set_label_pixmap(self.mask_image_label, pixmap, empty_text)

    def set_frame_text(self, text):
        self.frame_label.setText(text)

    def set_status_text(self, text, color="#888"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def set_navigation_enabled(self, enabled):
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.random_btn.setEnabled(enabled)


class ImageToSignalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image-to-Signal Processing Pipeline")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set window icon
        SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(SCRIPT_DIR, "app_icon.ico")
        print(f"Looking for icon at: {icon_path}")
        print(f"Icon exists: {os.path.isfile(icon_path)}")
        if os.path.isfile(icon_path):
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
            print(f"Icon loaded, null: {icon.isNull()}")
        
        # Calculate DATA_ROOT
        self.DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATA"))
        self.SYNTHETIC_DEFAULT_ROOT = os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "Simulation", "synthetic_data")
        )
        
        # Initialize config
        self.tool_id = 'tool'  # Start with 'tool' prefix
        self.config = self._create_default_config()
        self.mask_preview_dialog = None
        self.mask_preview_tool_id = None
        self.mask_preview_files = []
        self.mask_preview_index = 0
        self.mask_preview_background_cache = {}
        self.mask_preview_refresh_timer = QTimer(self)
        self.mask_preview_refresh_timer.setSingleShot(True)
        self.mask_preview_refresh_timer.timeout.connect(self._refresh_mask_preview)
        self._synthetic_mode_active = False
        self._synthetic_masks_dir = ""
        
        # Setup UI
        self._setup_ui()
        self._apply_styles()
        
    def _create_default_config(self):
        """Create default configuration dictionary."""
        return {
            'RAW_DIR': os.path.join(self.DATA_ROOT, 'tools', self.tool_id),
            'BLURRED_DIR': os.path.join(self.DATA_ROOT, 'blurred', f'{self.tool_id}_blurred'),
            'FINAL_MASKS_DIR': os.path.join(self.DATA_ROOT, 'masks', f'{self.tool_id}_final_masks'),
            'ROI_CSV_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_raw_data.csv'),
            'ROI_PLOT_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_raw_plot.svg'),
            'PROCESSED_CSV_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_processed_data.csv'),
            'PROCESSED_PLOT_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_processed_plot.svg'),
            'BACKGROUND_IMAGE_PATH': os.path.join(self.DATA_ROOT, 'backgrounds', 'paper_background.tiff'),
            'NUMBER_OF_PEAKS': 2,
            'blur_kernel': 13,
            'closing_kernel': 21,
            'h_threshold_min': 35,
            'h_threshold_max': 50,
            's_threshold_min': 38.25,
            's_threshold_max': 178.5,
            'V_threshold_min': 114.75,
            'V_threshold_max': 140.25,
            'L_threshold_min': 127.5,
            'L_threshold_max': 142.8,
            'a_threshold_min': 118,
            'a_threshold_max': 127,
            'b_threshold_min': 118,
            'b_threshold_max': 120,
            'BACKGROUND_SUBTRACTION_METHOD': 'lab',
            'APPLY_MULTICHANNEL_MASK': False,
            'DIFFERENCE_THRESHOLD': 33,
            'roi_height': 200,
            'analyze_full_picture': False,
            'WHITE_RATIO_OUTLIER_THRESHOLD': 0.8,
            'APPLY_MOVING_AVERAGE': True,
            'MOVING_AVERAGE_WINDOW': 5,
            'OPTIMIZATION_METHOD': 'gpu',  # 'gpu', 'multicore', or 'single' - applies to ALL steps
        }
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Tab widget for different sections
        tabs = QTabWidget()
        tabs.addTab(self._create_tool_config_tab(), "📁 Tool Configuration")
        tabs.addTab(self._create_processing_params_tab(), "⚙️ Processing Parameters")
        tabs.addTab(self._create_analysis_params_tab(), "📊 Analysis Parameters")
        tabs.addTab(self._create_pipeline_tab(), "▶️ Run Pipeline")
        tabs.addTab(self._create_360_utils_tab(), "🔄 360° Utilities")
        tabs.addTab(self._create_synthetic_masks_tab(), "🎭 Synthetic Dashboard")
        tabs.addTab(self._create_compare_tools_tab(), "⚖️ Compare Tools")
        main_layout.addWidget(tabs, stretch=1)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background: #2d2d2d; color: #4CAF50; border-radius: 3px;")
        main_layout.addWidget(self.status_label)
    
    def _create_header(self):
        """Create header with title, tool selector, and optimization selector."""
        header = QFrame()
        header.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(header)
        
        # Top row: Title and Tool ID
        top_row = QHBoxLayout()
        
        title = QLabel("🔧 Image-to-Signal Processing")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #4CAF50;")
        
        top_row.addWidget(title)
        top_row.addStretch()
        
        # Tool ID selector
        top_row.addWidget(QLabel("Tool ID:"))
        self.tool_id_input = QLineEdit(self.tool_id)
        self.tool_id_input.setMaximumWidth(200)
        self.tool_id_input.textChanged.connect(self._on_tool_id_changed)
        top_row.addWidget(self.tool_id_input)
        
        # Multi-select button
        multi_select_btn = QPushButton("📋 Select Multiple")
        multi_select_btn.setMaximumWidth(120)
        multi_select_btn.clicked.connect(self._open_tool_selector)
        top_row.addWidget(multi_select_btn)
        
        layout.addLayout(top_row)
        
        # Bottom row: Optimization selector (applies to all steps)
        opt_row = QHBoxLayout()
        
        opt_label = QLabel("🚀 Processing Mode:")
        opt_label.setStyleSheet("font-weight: bold;")
        opt_row.addWidget(opt_label)
        
        self.optimization_method = QComboBox()
        self.optimization_method.addItems(['gpu', 'multicore', 'single'])
        self.optimization_method.setCurrentText(self.config['OPTIMIZATION_METHOD'])
        self.optimization_method.currentTextChanged.connect(self._on_optimization_changed)
        self.optimization_method.setMaximumWidth(120)
        opt_row.addWidget(self.optimization_method)
        
        # GPU status indicator
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setStyleSheet("padding: 3px 8px; border-radius: 3px;")
        opt_row.addWidget(self.gpu_status_label)
        
        # Hardware info label
        self.hw_info_label = QLabel()
        self.hw_info_label.setStyleSheet("color: #888; font-size: 10px;")
        opt_row.addWidget(self.hw_info_label)
        
        opt_row.addStretch()
        
        # Help text
        help_text = QLabel("<i>⚠️ GPU mode works best with Intel Iris Xe. Applies to all pipeline steps.</i>")
        help_text.setStyleSheet("color: #888; font-size: 10px;")
        opt_row.addWidget(help_text)
        
        layout.addLayout(opt_row)
        
        # Initialize GPU status
        self._update_gpu_status()
        
        return header
    
    def _create_tool_config_tab(self):
        """Create tool configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Paths group
        paths_group = QGroupBox("File Paths")
        paths_layout = QVBoxLayout()
        
        self.path_labels = {}
        path_keys = ['RAW_DIR', 'BLURRED_DIR', 'FINAL_MASKS_DIR', 'ROI_CSV_PATH', 
                     'PROCESSED_CSV_PATH', 'BACKGROUND_IMAGE_PATH']
        
        for key in path_keys:
            row = QHBoxLayout()
            label = QLabel(key.replace('_', ' ').title() + ":")
            label.setMinimumWidth(200)
            path_label = QLabel(self.config[key])
            path_label.setStyleSheet("background: #1e1e1e; padding: 5px; border-radius: 3px; color: #888;")
            path_label.setWordWrap(True)
            self.path_labels[key] = path_label
            
            row.addWidget(label)
            row.addWidget(path_label, stretch=1)
            paths_layout.addLayout(row)
        
        paths_group.setLayout(paths_layout)
        scroll_layout.addWidget(paths_group)

        scroll_layout.addStretch()
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return widget
    
    def _create_processing_params_tab(self):
        """Create processing parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Image Processing
        img_group = QGroupBox("Image Processing")
        img_layout = QVBoxLayout()
        
        self.blur_kernel = self._add_spinbox_param(img_layout, "Blur Kernel:", 1, 99, 2, self.config['blur_kernel'])
        self.closing_kernel = self._add_spinbox_param(img_layout, "Closing Kernel:", 1, 99, 2, self.config['closing_kernel'])
        
        img_group.setLayout(img_layout)
        scroll_layout.addWidget(img_group)
        
        # Background Subtraction
        bg_group = QGroupBox("Background Subtraction")
        bg_layout = QVBoxLayout()
        
        # Info label explaining the methods
        info_label = QLabel(
            "<b>Methods:</b><br>"
            "• <b>none</b>: No background subtraction (relies on multichannel mask only)<br>"
            "• <b>absdiff</b>: Grayscale absolute difference (simple, fast)<br>"
            "• <b>lab</b>: LAB color space Delta E (perceptual color difference, sophisticated)<br><br>"
            "<b>Combination:</b> If both method + multichannel mask are enabled, results are combined (union)."
        )
        info_label.setStyleSheet("color: #888; font-size: 10px; padding: 5px; background: #1a1a1a; border-radius: 3px;")
        info_label.setWordWrap(True)
        bg_layout.addWidget(info_label)
        
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.bg_method = QComboBox()
        self.bg_method.addItems(['none', 'absdiff', 'lab'])
        self.bg_method.setCurrentText(self.config['BACKGROUND_SUBTRACTION_METHOD'])
        method_row.addWidget(self.bg_method)
        method_row.addStretch()
        bg_layout.addLayout(method_row)
        
        # Background image selector
        bg_image_row = QHBoxLayout()
        bg_image_row.addWidget(QLabel("Background Image:"))
        self.bg_image_combo = QComboBox()
        self.bg_image_combo.addItems(['paper_background.tiff', 'normal_background.tiff'])
        # Set current background based on config
        current_bg = os.path.basename(self.config.get('BACKGROUND_IMAGE_PATH', 'paper_background.tiff'))
        self.bg_image_combo.setCurrentText(current_bg)
        bg_image_row.addWidget(self.bg_image_combo)
        bg_image_row.addStretch()
        bg_layout.addLayout(bg_image_row)
        
        self.apply_multichannel = QCheckBox("Apply Multichannel Mask (uses HSV+LAB thresholds below)")
        self.apply_multichannel.setChecked(self.config['APPLY_MULTICHANNEL_MASK'])
        bg_layout.addWidget(self.apply_multichannel)
        
        self.diff_threshold = self._add_spinbox_param(bg_layout, "Difference Threshold:", 0, 255, 1, self.config['DIFFERENCE_THRESHOLD'])
        
        bg_group.setLayout(bg_layout)
        scroll_layout.addWidget(bg_group)
        
        # Threshold Parameters (HSV, LAB)
        thresh_group = QGroupBox("Color Thresholds")
        thresh_layout = QVBoxLayout()
        
        # HSV
        hsv_label = QLabel("HSV Thresholds")
        hsv_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        thresh_layout.addWidget(hsv_label)
        self.h_min = self._add_double_spinbox_param(thresh_layout, "H Min:", 0, 180, self.config['h_threshold_min'])
        self.h_max = self._add_double_spinbox_param(thresh_layout, "H Max:", 0, 180, self.config['h_threshold_max'])
        self.s_min = self._add_double_spinbox_param(thresh_layout, "S Min:", 0, 255, self.config['s_threshold_min'])
        self.s_max = self._add_double_spinbox_param(thresh_layout, "S Max:", 0, 255, self.config['s_threshold_max'])
        self.v_min = self._add_double_spinbox_param(thresh_layout, "V Min:", 0, 255, self.config['V_threshold_min'])
        self.v_max = self._add_double_spinbox_param(thresh_layout, "V Max:", 0, 255, self.config['V_threshold_max'])
        
        # LAB
        lab_label = QLabel("LAB Thresholds")
        lab_label.setStyleSheet("font-weight: bold; color: #4CAF50; margin-top: 15px;")
        thresh_layout.addWidget(lab_label)
        self.l_min = self._add_double_spinbox_param(thresh_layout, "L Min:", 0, 255, self.config['L_threshold_min'])
        self.l_max = self._add_double_spinbox_param(thresh_layout, "L Max:", 0, 255, self.config['L_threshold_max'])
        self.a_min = self._add_double_spinbox_param(thresh_layout, "a Min:", 0, 255, self.config['a_threshold_min'])
        self.a_max = self._add_double_spinbox_param(thresh_layout, "a Max:", 0, 255, self.config['a_threshold_max'])
        self.b_min = self._add_double_spinbox_param(thresh_layout, "b Min:", 0, 255, self.config['b_threshold_min'])
        self.b_max = self._add_double_spinbox_param(thresh_layout, "b Max:", 0, 255, self.config['b_threshold_max'])
        
        thresh_group.setLayout(thresh_layout)
        scroll_layout.addWidget(thresh_group)

        preview_row = QHBoxLayout()
        preview_hint = QLabel(
            "Preview step 2 output on the current blurred frames. While the preview window is open, "
            "changes to mask-generation parameters refresh automatically."
        )
        preview_hint.setWordWrap(True)
        preview_hint.setStyleSheet("color: #888; font-size: 10px;")
        preview_row.addWidget(preview_hint, stretch=1)

        self.preview_mask_btn = QPushButton("👁️ Preview Mask")
        self.preview_mask_btn.clicked.connect(self._open_mask_preview)
        preview_row.addWidget(self.preview_mask_btn)
        scroll_layout.addLayout(preview_row)

        self._connect_mask_preview_controls()
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return widget
    
    def _create_analysis_params_tab(self):
        """Create analysis parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # ROI Analysis
        roi_group = QGroupBox("ROI Analysis")
        roi_layout = QVBoxLayout()
        
        self.roi_height = self._add_spinbox_param(roi_layout, "ROI Height:", 1, 1000, 1, self.config['roi_height'])
        
        self.analyze_full_picture = QCheckBox("Analyze Full Picture (Ignore ROI Height)")
        self.analyze_full_picture.setChecked(self.config.get('analyze_full_picture', False))
        roi_layout.addWidget(self.analyze_full_picture)
        
        self.white_ratio_threshold = self._add_double_spinbox_param(roi_layout, "White Ratio Outlier Threshold:", 0.0, 1.0, self.config['WHITE_RATIO_OUTLIER_THRESHOLD'])
        
        self.apply_moving_avg = QCheckBox("Apply Moving Average")
        self.apply_moving_avg.setChecked(self.config['APPLY_MOVING_AVERAGE'])
        roi_layout.addWidget(self.apply_moving_avg)
        
        self.moving_avg_window = self._add_spinbox_param(roi_layout, "Moving Average Window:", 1, 50, 1, self.config['MOVING_AVERAGE_WINDOW'])
        
        roi_group.setLayout(roi_layout)
        scroll_layout.addWidget(roi_group)
        
        # Processing
        proc_group = QGroupBox("Data Processing")
        proc_layout = QVBoxLayout()
        
        self.num_peaks = self._add_spinbox_param(proc_layout, "Number of Peaks:", 1, 10, 1, self.config['NUMBER_OF_PEAKS'])
        
        proc_group.setLayout(proc_layout)
        scroll_layout.addWidget(proc_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return widget
    
    def _create_pipeline_tab(self):
        """Create pipeline execution tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Step controls
        steps_group = QGroupBox("Pipeline Steps")
        steps_layout = QVBoxLayout()
        
        # Renaming moved to separate utility; update label accordingly
        self.step1_check = QCheckBox("Step 1: Blur Images")
        self.step2_check = QCheckBox("Step 2: Generate Masks")
        self.step3_check = QCheckBox("Step 3: Analyze ROI and Plot (Raw)")
        self.step4_check = QCheckBox("Step 4: Process and Plot (Normalized & Segmented)")
        
        self.step4_check.setChecked(True)  # Default
        
        steps_layout.addWidget(self.step1_check)
        steps_layout.addWidget(self.step2_check)
        steps_layout.addWidget(self.step3_check)
        steps_layout.addWidget(self.step4_check)
        
        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶️ Run Selected Steps")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #45a049;
            }
            QPushButton:disabled {
                background: #555;
                color: #888;
            }
        """)
        self.run_btn.clicked.connect(self._run_pipeline)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #da190b;
            }
            QPushButton:disabled {
                background: #555;
                color: #888;
            }
        """)
        self.stop_btn.clicked.connect(self._stop_pipeline)
        
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # Log output
        log_label = QLabel("Processing Log:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background: #1e1e1e; color: #ddd; font-family: 'Consolas', monospace;")
        layout.addWidget(self.log_output, stretch=1)
        
        # Quick open buttons
        open_label = QLabel("Quick Open:")
        open_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(open_label)
        
        open_layout = QHBoxLayout()
        
        self.open_raw_plot_btn = QPushButton("📊 Raw Plot")
        self.open_raw_plot_btn.clicked.connect(lambda: self._open_file('ROI_PLOT_PATH'))
        open_layout.addWidget(self.open_raw_plot_btn)
        
        self.open_processed_plot_btn = QPushButton("📈 Processed Plot")
        self.open_processed_plot_btn.clicked.connect(lambda: self._open_file('PROCESSED_PLOT_PATH'))
        open_layout.addWidget(self.open_processed_plot_btn)
        
        self.open_masks_btn = QPushButton("🎭 Masks Folder")
        self.open_masks_btn.clicked.connect(lambda: self._open_file('FINAL_MASKS_DIR'))
        open_layout.addWidget(self.open_masks_btn)
        
        self.open_blurred_btn = QPushButton("🌫️ Blurred Folder")
        self.open_blurred_btn.clicked.connect(lambda: self._open_file('BLURRED_DIR'))
        open_layout.addWidget(self.open_blurred_btn)
        
        self.open_raw_btn = QPushButton("📁 Raw Folder")
        self.open_raw_btn.clicked.connect(lambda: self._open_file('RAW_DIR'))
        open_layout.addWidget(self.open_raw_btn)
        
        layout.addLayout(open_layout)
        
        return widget

    def _create_360_utils_tab(self):
        """Create dedicated tab for 360° detection and renaming."""
        # Create scroll area to contain everything
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        info = QLabel("Tools to detect 360° frame count, remove extra images, and rename frames based on angles.")
        info.setStyleSheet("color: #ccc; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Three columns in one row for the controls
        controls_row = QHBoxLayout()
        controls_row.setSpacing(15)
        
        # First column: Detection
        detect_group = QGroupBox("1. Find 360° Frame Count")
        dg_layout = QVBoxLayout()
        find_btn = QPushButton("🔍 Run Similarity Detector\n(find360)")
        find_btn.setMinimumHeight(60)
        find_btn.setMaximumHeight(80)
        find_btn.setStyleSheet("font-size: 13px; font-weight: bold;")
        find_btn.clicked.connect(self._run_find360)
        dg_layout.addWidget(find_btn)
        dg_layout.addStretch()
        detect_group.setLayout(dg_layout)
        controls_row.addWidget(detect_group, 1)

        # Second column: Remove extra images
        remove_group = QGroupBox("2. Remove Extra Images")
        remove_layout = QVBoxLayout()
        self.remove_extra_label = QLabel("Run detector first to see how many extra images need removal.")
        self.remove_extra_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self.remove_extra_label.setWordWrap(True)
        self.remove_extra_label.setMinimumHeight(40)
        remove_layout.addWidget(self.remove_extra_label)
        self.remove_extra_btn = QPushButton("🗑️ Remove Extra Images")
        self.remove_extra_btn.setMinimumHeight(40)
        self.remove_extra_btn.setStyleSheet("font-size: 13px; font-weight: bold; background: #f44336;")
        self.remove_extra_btn.setEnabled(False)
        self.remove_extra_btn.clicked.connect(self._remove_extra_images)
        remove_layout.addWidget(self.remove_extra_btn)
        remove_layout.addStretch()
        remove_group.setLayout(remove_layout)
        controls_row.addWidget(remove_group, 1)

        # Third column: Rename
        rename_group = QGroupBox("3. Rename All Images by Angle")
        rg_layout = QVBoxLayout()
        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Frames for 360°:"))
        self.frames360_spin = QSpinBox()
        self.frames360_spin.setRange(1, 5000)
        self.frames360_spin.setValue(360)
        self.frames360_spin.setMinimumWidth(100)
        self.frames360_spin.setSpecialValueText("Auto-detect first")
        frames_row.addWidget(self.frames360_spin)
        frames_row.addStretch()
        rg_layout.addLayout(frames_row)
        rename_btn = QPushButton("📝 Rename All Images\n(Masks + Blurred)")
        rename_btn.setMinimumHeight(40)
        rename_btn.setStyleSheet("font-size: 13px; font-weight: bold;")
        rename_btn.clicked.connect(self._run_rename_all)
        rg_layout.addWidget(rename_btn)
        rg_layout.addStretch()
        rename_group.setLayout(rg_layout)
        controls_row.addWidget(rename_group, 1)
        
        layout.addLayout(controls_row)

        # Terminal output with larger height
        log_label = QLabel("Terminal Output:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)
        
        self.utils_log_output = QTextEdit()
        self.utils_log_output.setReadOnly(True)
        self.utils_log_output.setStyleSheet("background:#1e1e1e; color:#ccc; font-family:Consolas; font-size: 11px;")
        self.utils_log_output.setMinimumHeight(200)
        self.utils_log_output.setMaximumHeight(250)
        layout.addWidget(self.utils_log_output)

        # Plot section with better sizing
        plot_label = QLabel("360° Detection Plot:")
        plot_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(plot_label)
        
        # Matplotlib embedded figure with larger size
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        self.find360_fig = Figure(figsize=(11, 6.5))
        self.find360_canvas = FigureCanvas(self.find360_fig)
        self.find360_toolbar = NavigationToolbar(self.find360_canvas, scroll_content)
        
        layout.addWidget(self.find360_toolbar)
        layout.addWidget(self.find360_canvas)
        
        layout.addStretch()
        
        scroll.setWidget(scroll_content)
        return scroll
    
    def _create_synthetic_masks_tab(self):
        """Create dedicated tab for Synthetic Masks Dashboard."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        info = QLabel("Analyze synthetic mask images directly and generate an interactive HTML dashboard.")
        info.setStyleSheet("color: #ccc; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Synthetic Path Selector
        synthetic_group = QGroupBox("Synthetic Data Paths")
        synthetic_layout = QVBoxLayout()
        
        synthetic_path_row = QHBoxLayout()
        synthetic_path_row.addWidget(QLabel("Masks Folder (Req):"))
        self.synthetic_masks_dir_input = QLineEdit(self.SYNTHETIC_DEFAULT_ROOT)
        synthetic_path_row.addWidget(self.synthetic_masks_dir_input, stretch=1)
        synthetic_browse_btn = QPushButton("Browse...")
        synthetic_browse_btn.clicked.connect(self._browse_synthetic_masks_dir)
        synthetic_path_row.addWidget(synthetic_browse_btn)
        synthetic_layout.addLayout(synthetic_path_row)
        
        real_path_row = QHBoxLayout()
        real_path_row.addWidget(QLabel("Realistic Folder (Opt):"))
        self.synthetic_real_dir_input = QLineEdit()
        real_path_row.addWidget(self.synthetic_real_dir_input, stretch=1)
        real_browse_btn = QPushButton("Browse...")
        real_browse_btn.clicked.connect(self._browse_synthetic_real_dir)
        real_path_row.addWidget(real_browse_btn)
        synthetic_layout.addLayout(real_path_row)
        
        csv_path_row = QHBoxLayout()
        csv_path_row.addWidget(QLabel("CSV File (Opt):"))
        self.synthetic_csv_input = QLineEdit()
        csv_path_row.addWidget(self.synthetic_csv_input, stretch=1)
        csv_browse_btn = QPushButton("Browse...")
        csv_browse_btn.clicked.connect(self._browse_synthetic_csv)
        csv_path_row.addWidget(csv_browse_btn)
        synthetic_layout.addLayout(csv_path_row)
        
        synthetic_group.setLayout(synthetic_layout)
        layout.addWidget(synthetic_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.run_synthetic_btn = QPushButton("▶️ Run Analysis & Generate Dashboard")
        self.run_synthetic_btn.setMinimumHeight(40)
        self.run_synthetic_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        self.run_synthetic_btn.clicked.connect(self._run_synthetic_pipeline_and_dashboard)
        actions_layout.addWidget(self.run_synthetic_btn)
        
        self.open_dashboard_btn = QPushButton("🌐 Open Existing Dashboard")
        self.open_dashboard_btn.setMinimumHeight(40)
        self.open_dashboard_btn.clicked.connect(self._open_existing_dashboard)
        actions_layout.addWidget(self.open_dashboard_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        return widget

    def _create_compare_tools_tab(self):
        """Create tab for comparing two tools side-by-side."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        info = QLabel("Compare two tools mathematically and visually. Select the Masks folders for both tools.")
        info.setStyleSheet("color: #ccc; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Tool A
        group_A = QGroupBox("Tool A (e.g. Healthy Tool)")
        layout_A = QVBoxLayout()
        row_A = QHBoxLayout()
        self.compare_A_input = QLineEdit(self.SYNTHETIC_DEFAULT_ROOT)
        row_A.addWidget(self.compare_A_input, stretch=1)
        btn_A = QPushButton("Browse...")
        btn_A.clicked.connect(self._browse_compare_A)
        row_A.addWidget(btn_A)
        layout_A.addLayout(row_A)
        group_A.setLayout(layout_A)
        layout.addWidget(group_A)
        
        # Tool B
        group_B = QGroupBox("Tool B (e.g. Broken Tool)")
        layout_B = QVBoxLayout()
        row_B = QHBoxLayout()
        self.compare_B_input = QLineEdit(self.SYNTHETIC_DEFAULT_ROOT)
        row_B.addWidget(self.compare_B_input, stretch=1)
        btn_B = QPushButton("Browse...")
        btn_B.clicked.connect(self._browse_compare_B)
        row_B.addWidget(btn_B)
        layout_B.addLayout(row_B)
        group_B.setLayout(layout_B)
        layout.addWidget(group_B)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        self.run_compare_btn = QPushButton("⚖️ Generate Comparison Dashboard")
        self.run_compare_btn.setMinimumHeight(50)
        self.run_compare_btn.setStyleSheet("""
            QPushButton {
                background: #FFB74D;
                color: #333;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background: #FFA726; }
        """)
        self.run_compare_btn.clicked.connect(self._run_compare_dashboard)
        actions_layout.addWidget(self.run_compare_btn)
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        return widget
    
    def _add_spinbox_param(self, layout, label, min_val, max_val, step, default):
        """Helper to add spinbox parameter."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(200)
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(default)
        spinbox.setMaximumWidth(100)
        row.addWidget(lbl)
        row.addWidget(spinbox)
        row.addStretch()
        layout.addLayout(row)
        return spinbox
    
    def _add_double_spinbox_param(self, layout, label, min_val, max_val, default):
        """Helper to add double spinbox parameter."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(200)
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setDecimals(2)
        spinbox.setValue(default)
        spinbox.setMaximumWidth(100)
        row.addWidget(lbl)
        row.addWidget(spinbox)
        row.addStretch()
        layout.addLayout(row)
        return spinbox

    def _connect_mask_preview_controls(self):
        """Refresh the preview window whenever mask-generation controls change."""
        preview_controls = [
            (self.tool_id_input, 'textChanged'),
            (self.closing_kernel, 'valueChanged'),
            (self.bg_method, 'currentTextChanged'),
            (self.bg_image_combo, 'currentTextChanged'),
            (self.apply_multichannel, 'stateChanged'),
            (self.diff_threshold, 'valueChanged'),
            (self.h_min, 'valueChanged'),
            (self.h_max, 'valueChanged'),
            (self.s_min, 'valueChanged'),
            (self.s_max, 'valueChanged'),
            (self.v_min, 'valueChanged'),
            (self.v_max, 'valueChanged'),
            (self.l_min, 'valueChanged'),
            (self.l_max, 'valueChanged'),
            (self.a_min, 'valueChanged'),
            (self.a_max, 'valueChanged'),
            (self.b_min, 'valueChanged'),
            (self.b_max, 'valueChanged'),
        ]

        for widget, signal_name in preview_controls:
            signal = getattr(widget, signal_name)
            signal.connect(lambda *_: self._schedule_mask_preview_refresh())

    def _schedule_mask_preview_refresh(self):
        """Debounce mask preview refresh while the dialog is open."""
        if self.mask_preview_dialog is None or not self.mask_preview_dialog.isVisible():
            return
        self.mask_preview_refresh_timer.start(120)

    def _get_primary_tool_id(self):
        """Return the first tool id from the current selector text."""
        tool_ids = [tool.strip() for tool in self.tool_id_input.text().split(',') if tool.strip()]
        return tool_ids[0] if tool_ids else ""

    def _list_preview_source_files(self, folder_path):
        """List source frames available for preview."""
        valid_exts = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')
        return sorted([name for name in os.listdir(folder_path) if name.lower().endswith(valid_exts)])

    def _load_preview_background_image(self, background_path):
        """Load and cache the selected background image for repeated preview refreshes."""
        cached = self.mask_preview_background_cache.get(background_path)
        if cached is not None:
            return cached

        with Image.open(background_path) as bg_image:
            background_np = np.array(bg_image.convert('RGB'))

        self.mask_preview_background_cache[background_path] = background_np
        return background_np

    def _build_mask_preview_arrays(self, image_path, config):
        """Generate the final step 2 mask for a single blurred frame."""
        with Image.open(image_path) as image_file:
            blurred_image = image_file.convert('RGB')

        original_np = np.array(blurred_image)
        method = config.get('BACKGROUND_SUBTRACTION_METHOD', 'none').lower()
        use_mc_mask = config.get('APPLY_MULTICHANNEL_MASK', False)

        bg_mask_np = None
        color_mask_np = None

        if method in ('absdiff', 'lab'):
            background_path = config.get('BACKGROUND_IMAGE_PATH', '')
            if not os.path.isfile(background_path):
                return original_np, None, f"Background image not found: {background_path}"

            background_image_np = self._load_preview_background_image(background_path)
            if method == 'absdiff':
                bg_mask = background_subtraction_absdiff(blurred_image, background_image_np, config)
            else:
                bg_mask = background_subtraction_lab(blurred_image, background_image_np, config)

            if bg_mask is not None:
                bg_mask_np = np.array(bg_mask.convert('L'))

        if use_mc_mask:
            color_mask = create_multichannel_mask(blurred_image, config)
            if color_mask is not None:
                color_mask_np = np.array(color_mask.convert('L'))

        if bg_mask_np is not None and color_mask_np is not None:
            initial_mask_np = cv2.bitwise_or(bg_mask_np, color_mask_np)
        elif bg_mask_np is not None:
            initial_mask_np = bg_mask_np
        elif color_mask_np is not None:
            initial_mask_np = color_mask_np
        else:
            return original_np, None, "Enable background subtraction or multichannel masking to preview a mask."

        initial_mask = Image.fromarray(initial_mask_np)
        filled1 = fill_holes(initial_mask)
        closed = morph_closing(filled1, kernel_size=config.get('closing_kernel', 21)) if filled1 else None
        largest_contour = keep_largest_contour(closed) if closed else None
        filled2 = fill_holes(largest_contour) if largest_contour else None

        if filled2 is None:
            return original_np, None, "Mask post-processing failed for this frame."

        return original_np, np.array(filled2.convert('L')), None

    def _numpy_to_pixmap(self, image_np):
        """Convert a numpy image array to QPixmap for preview display."""
        if image_np is None:
            return None

        if image_np.ndim == 2:
            grayscale = np.ascontiguousarray(image_np.astype(np.uint8))
            qimage = QImage(
                grayscale.data,
                grayscale.shape[1],
                grayscale.shape[0],
                grayscale.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(qimage)

        rgb = np.ascontiguousarray(image_np.astype(np.uint8))
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimage)

    def _open_mask_preview(self):
        """Open the modeless mask preview dialog."""
        preview_tool = self._get_primary_tool_id()
        if not preview_tool:
            QMessageBox.warning(self, "Preview Unavailable", "Enter a tool ID before opening the mask preview.")
            return

        self._update_config_from_ui()
        preview_config = self._build_config_for_tool(preview_tool)
        try:
            preview_files = self._list_preview_source_files(preview_config['BLURRED_DIR'])
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "Preview Unavailable",
                f"Blurred folder not found for {preview_tool}:\n{preview_config['BLURRED_DIR']}\n\nRun step 1 first.",
            )
            return

        if not preview_files:
            QMessageBox.warning(
                self,
                "Preview Unavailable",
                f"No blurred frames found for {preview_tool}:\n{preview_config['BLURRED_DIR']}\n\nRun step 1 first.",
            )
            return

        if self.mask_preview_dialog is None:
            self.mask_preview_dialog = MaskPreviewDialog(self)
            self.mask_preview_dialog.previous_requested.connect(self._show_previous_mask_preview_frame)
            self.mask_preview_dialog.next_requested.connect(self._show_next_mask_preview_frame)
            self.mask_preview_dialog.random_requested.connect(self._show_random_mask_preview_frame)
            self.mask_preview_dialog.finished.connect(self._on_mask_preview_closed)

        self.mask_preview_tool_id = preview_tool
        self.mask_preview_files = preview_files
        self.mask_preview_index = 0
        self.mask_preview_dialog.setWindowTitle(f"Mask Preview - {preview_tool}")
        self.mask_preview_dialog.show()
        self.mask_preview_dialog.raise_()
        self.mask_preview_dialog.activateWindow()
        self._refresh_mask_preview()

    def _show_previous_mask_preview_frame(self):
        """Navigate to the previous preview frame."""
        if not self.mask_preview_files:
            return
        self.mask_preview_index = (self.mask_preview_index - 1) % len(self.mask_preview_files)
        self._refresh_mask_preview()

    def _show_next_mask_preview_frame(self):
        """Navigate to the next preview frame."""
        if not self.mask_preview_files:
            return
        self.mask_preview_index = (self.mask_preview_index + 1) % len(self.mask_preview_files)
        self._refresh_mask_preview()

    def _show_random_mask_preview_frame(self):
        """Jump to a random preview frame."""
        if not self.mask_preview_files:
            return
        if len(self.mask_preview_files) == 1:
            self.mask_preview_index = 0
        else:
            available_indices = [idx for idx in range(len(self.mask_preview_files)) if idx != self.mask_preview_index]
            self.mask_preview_index = random.choice(available_indices)
        self._refresh_mask_preview()

    def _refresh_mask_preview(self):
        """Refresh the preview dialog using the current parameters and frame selection."""
        if self.mask_preview_dialog is None or not self.mask_preview_dialog.isVisible():
            return

        preview_tool = self._get_primary_tool_id()
        if not preview_tool:
            self.mask_preview_dialog.set_frame_text("Frame: -")
            self.mask_preview_dialog.set_original_preview(None, "No tool selected")
            self.mask_preview_dialog.set_mask_preview(None, "No tool selected")
            self.mask_preview_dialog.set_status_text("Enter a tool ID to preview a mask.", "#FFA726")
            self.mask_preview_dialog.set_navigation_enabled(False)
            return

        self._update_config_from_ui()
        preview_config = self._build_config_for_tool(preview_tool)

        try:
            preview_files = self._list_preview_source_files(preview_config['BLURRED_DIR'])
        except FileNotFoundError:
            preview_files = []

        if preview_tool != self.mask_preview_tool_id:
            self.mask_preview_tool_id = preview_tool
            self.mask_preview_index = 0

        self.mask_preview_files = preview_files
        self.mask_preview_dialog.setWindowTitle(f"Mask Preview - {preview_tool}")

        if not preview_files:
            self.mask_preview_dialog.set_frame_text(f"Tool {preview_tool} | No blurred frames found")
            self.mask_preview_dialog.set_original_preview(None, "No blurred frames found")
            self.mask_preview_dialog.set_mask_preview(None, "No mask preview available")
            self.mask_preview_dialog.set_status_text("Run step 1 first to generate blurred frames for preview.", "#FFA726")
            self.mask_preview_dialog.set_navigation_enabled(False)
            return

        self.mask_preview_index = max(0, min(self.mask_preview_index, len(preview_files) - 1))
        current_filename = preview_files[self.mask_preview_index]
        current_path = os.path.join(preview_config['BLURRED_DIR'], current_filename)
        original_np, mask_np, error_message = self._build_mask_preview_arrays(current_path, preview_config)

        self.mask_preview_dialog.set_original_preview(self._numpy_to_pixmap(original_np), "Could not load source frame")
        self.mask_preview_dialog.set_mask_preview(
            self._numpy_to_pixmap(mask_np),
            "Mask preview unavailable for this frame",
        )
        self.mask_preview_dialog.set_frame_text(
            f"Tool {preview_tool} | Frame {self.mask_preview_index + 1}/{len(preview_files)} | {current_filename}"
        )
        self.mask_preview_dialog.set_navigation_enabled(len(preview_files) > 0)

        if error_message:
            self.mask_preview_dialog.set_status_text(error_message, "#FFA726")
            return

        white_ratio = float(np.mean(mask_np == 255)) if mask_np is not None and mask_np.size else 0.0
        method_label = preview_config.get('BACKGROUND_SUBTRACTION_METHOD', 'none')
        multichannel_label = "on" if preview_config.get('APPLY_MULTICHANNEL_MASK', False) else "off"
        self.mask_preview_dialog.set_status_text(
            f"Preview refreshed. Method: {method_label}; multichannel: {multichannel_label}; white ratio: {white_ratio:.3f}",
            "#4CAF50",
        )

    def _on_mask_preview_closed(self):
        """Clean up preview dialog state when the window closes."""
        self.mask_preview_refresh_timer.stop()
        self.mask_preview_dialog = None
        self.mask_preview_files = []
        self.mask_preview_tool_id = None
        self.mask_preview_index = 0
    
    def _on_tool_id_changed(self):
        """Update paths when tool ID changes."""
        self.tool_id = self.tool_id_input.text()
        self.config = self._create_default_config()
        
        # Update path labels
        for key, label in self.path_labels.items():
            label.setText(self.config[key])

        self._schedule_mask_preview_refresh()
    
    def _on_optimization_changed(self, method):
        """Handle optimization method change."""
        self._update_gpu_status()
        if method == 'gpu':
            self.status_label.setText("GPU acceleration selected (Intel Iris Xe recommended)")
            self.status_label.setStyleSheet("padding: 5px; background: #2d2d2d; color: #4CAF50; border-radius: 3px;")
        elif method == 'multicore':
            import os
            cores = os.cpu_count() or 1
            self.status_label.setText(f"Multi-core processing selected ({cores} cores available)")
            self.status_label.setStyleSheet("padding: 5px; background: #2d2d2d; color: #9b59b6; border-radius: 3px;")
        else:
            self.status_label.setText("Single-core processing selected (slowest)")
            self.status_label.setStyleSheet("padding: 5px; background: #2d2d2d; color: #e67e22; border-radius: 3px;")
    
    def _update_gpu_status(self):
        """Check and display GPU availability status."""
        try:
            from .utils.optimized_processing import check_gpu_available, get_optimization_info
            gpu_available, device_name, message = check_gpu_available()
            info = get_optimization_info()
            
            if gpu_available:
                is_intel = info['gpu'].get('is_intel', False)
                if is_intel:
                    self.gpu_status_label.setText(f"✅ {device_name}")
                    self.gpu_status_label.setStyleSheet("padding: 3px 8px; background: #1a3a1a; color: #4CAF50; border-radius: 3px;")
                    self.hw_info_label.setText(f"Intel GPU Ready")
                else:
                    self.gpu_status_label.setText(f"⚠️ {device_name}")
                    self.gpu_status_label.setStyleSheet("padding: 3px 8px; background: #3a3a1a; color: #FFA500; border-radius: 3px;")
                    self.hw_info_label.setText("Non-Intel GPU (may not be optimal)")
            else:
                self.gpu_status_label.setText("❌ No GPU")
                self.gpu_status_label.setStyleSheet("padding: 3px 8px; background: #3a1a1a; color: #e74c3c; border-radius: 3px;")
                self.hw_info_label.setText("Will fallback to multi-core if GPU selected")
        except Exception as e:
            self.gpu_status_label.setText("⚠️ Unknown")
            self.gpu_status_label.setStyleSheet("padding: 3px 8px; background: #3a3a1a; color: #FFA500; border-radius: 3px;")
            self.hw_info_label.setText(f"Could not check GPU: {str(e)[:30]}")

    def _update_config_from_ui(self):
        """Update config dictionary from UI values."""
        self.config['blur_kernel'] = self.blur_kernel.value()
        self.config['closing_kernel'] = self.closing_kernel.value()
        self.config['OPTIMIZATION_METHOD'] = self.optimization_method.currentText()
        self.config['BACKGROUND_SUBTRACTION_METHOD'] = self.bg_method.currentText()
        
        # Update background image path based on selection
        bg_filename = self.bg_image_combo.currentText()
        self.config['BACKGROUND_IMAGE_PATH'] = os.path.join(self.DATA_ROOT, 'backgrounds', bg_filename)
        
        self.config['APPLY_MULTICHANNEL_MASK'] = self.apply_multichannel.isChecked()
        self.config['DIFFERENCE_THRESHOLD'] = self.diff_threshold.value()
        
        # Thresholds
        self.config['h_threshold_min'] = self.h_min.value()
        self.config['h_threshold_max'] = self.h_max.value()
        self.config['s_threshold_min'] = self.s_min.value()
        self.config['s_threshold_max'] = self.s_max.value()
        self.config['V_threshold_min'] = self.v_min.value()
        self.config['V_threshold_max'] = self.v_max.value()
        self.config['L_threshold_min'] = self.l_min.value()
        self.config['L_threshold_max'] = self.l_max.value()
        self.config['a_threshold_min'] = self.a_min.value()
        self.config['a_threshold_max'] = self.a_max.value()
        self.config['b_threshold_min'] = self.b_min.value()
        self.config['b_threshold_max'] = self.b_max.value()
        
        # Analysis
        self.config['analyze_full_picture'] = self.analyze_full_picture.isChecked()
        if self.config['analyze_full_picture']:
            self.config['roi_height'] = 0
        else:
            self.config['roi_height'] = self.roi_height.value()
            
        self.config['WHITE_RATIO_OUTLIER_THRESHOLD'] = self.white_ratio_threshold.value()
        self.config['APPLY_MOVING_AVERAGE'] = self.apply_moving_avg.isChecked()
        self.config['MOVING_AVERAGE_WINDOW'] = self.moving_avg_window.value()
        self.config['NUMBER_OF_PEAKS'] = self.num_peaks.value()
    
    def _build_config_for_tool(self, tool_id):
        """Build a complete config dictionary for a specific tool ID."""
        config = self.config.copy()
        config['tool_id'] = tool_id
        
        # Update paths for this specific tool
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "DATA")
        config['DATA_ROOT'] = data_root
        config['RAW_DIR'] = os.path.join(data_root, 'tools', tool_id)
        config['BLURRED_DIR'] = os.path.join(data_root, 'blurred', f'{tool_id}_blurred')
        config['FINAL_MASKS_DIR'] = os.path.join(data_root, 'masks', f'{tool_id}_final_masks')
        # Standardized naming convention for outputs
        config['ROI_CSV_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_raw_data.csv')
        config['ROI_PLOT_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_raw_plot.svg')
        config['PROCESSED_CSV_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_processed_data.csv')
        config['PROCESSED_PLOT_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_processed_plot.svg')
        
        return config

    def _browse_synthetic_masks_dir(self):
        """Select an external folder containing synthetic mask images."""
        start_dir = self.synthetic_masks_dir_input.text().strip() or self.SYNTHETIC_DEFAULT_ROOT
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Synthetic Masks Folder",
            start_dir,
        )
        if selected_dir:
            self.synthetic_masks_dir_input.setText(selected_dir)
            
    def _browse_synthetic_real_dir(self):
        """Select an external folder containing synthetic realistic images."""
        start_dir = self.synthetic_real_dir_input.text().strip() or self.SYNTHETIC_DEFAULT_ROOT
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Synthetic Realistic Folder",
            start_dir,
        )
        if selected_dir:
            self.synthetic_real_dir_input.setText(selected_dir)
            
    def _browse_synthetic_csv(self):
        """Select a specific CSV file."""
        start_dir = self.synthetic_masks_dir_input.text().strip() or self.SYNTHETIC_DEFAULT_ROOT
        selected_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            start_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        if selected_file:
            self.synthetic_csv_input.setText(selected_file)

    def _build_config_for_synthetic_masks(self, masks_dir, tool_id):
        """Build config for external synthetic masks where only Step 3/4 is needed."""
        config = self.config.copy()
        masks_dir = os.path.abspath(masks_dir)
        tool_id = tool_id or "synthetic_masks"
        analysis_dir = os.path.join(masks_dir, "analysis")

        config['tool_id'] = tool_id
        config['DATA_ROOT'] = os.path.dirname(masks_dir)
        config['RAW_DIR'] = masks_dir
        config['BLURRED_DIR'] = masks_dir
        config['FINAL_MASKS_DIR'] = masks_dir
        config['ANALYSIS_OUTPUT_DIR'] = analysis_dir
        config['ROI_CSV_PATH'] = os.path.join(analysis_dir, f'{tool_id}_raw_data.csv')
        config['ROI_PLOT_PATH'] = os.path.join(analysis_dir, f'{tool_id}_raw_plot.svg')
        config['PROCESSED_CSV_PATH'] = os.path.join(analysis_dir, f'{tool_id}_processed_data.csv')
        config['PROCESSED_PLOT_PATH'] = os.path.join(analysis_dir, f'{tool_id}_processed_plot.svg')
        return config
    
    def _run_pipeline(self):
        """Run selected pipeline steps."""
        self._update_config_from_ui()
        self._synthetic_mode_active = False
        self._generate_dashboard_after = False
        
        steps = []
        if self.step1_check.isChecked():
            steps.append(("Step 1: Blur and Rename", step1_blur_and_rename.run))
        if self.step2_check.isChecked():
            steps.append(("Step 2: Generate Masks", step2_generate_masks.run))
        if self.step3_check.isChecked():
            steps.append(("Step 3: Analyze and Plot", step3_analyze_and_plot.run))
        if self.step4_check.isChecked():
            steps.append(("Step 4: Process and Plot", step4_process_and_plot.run))
        
        if not steps:
            self.log_output.append("<span style='color: orange;'>⚠️ No steps selected!</span>")
            return

        # Parse tool IDs (comma-separated)
        tool_ids = [t.strip() for t in self.tool_id.split(',') if t.strip()]
        if not tool_ids:
            self.log_output.append("<span style='color: orange;'>⚠️ No tool ID specified!</span>")
            return
        
        self.log_output.clear()
        if len(tool_ids) == 1:
            self.log_output.append(f"<span style='color: #4CAF50;'>🚀 Starting pipeline for {tool_ids[0]}...</span>")
        else:
            self.log_output.append(f"<span style='color: #4CAF50;'>🚀 Starting pipeline for {len(tool_ids)} tools...</span>")
            self.log_output.append(f"<span style='color: #888;'>Tools: {', '.join(tool_ids)}</span>")
        self.log_output.append(f"<span style='color: #888;'>Running {len(steps)} step(s) per tool</span>")
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Run steps sequentially for each tool
        self.current_step = 0
        self.current_tool_index = 0
        self.tools_to_process = tool_ids
        self.steps_to_run = steps
        
        # Build config for first tool
        self.config = self._build_config_for_tool(tool_ids[0])
        
        self._run_next_step()
        
    def _run_synthetic_pipeline_and_dashboard(self):
        """Run step 3 & 4 for synthetic masks and then generate dashboard."""
        self._update_config_from_ui()
        synthetic_masks_dir = self.synthetic_masks_dir_input.text().strip()
        if not synthetic_masks_dir or not os.path.isdir(synthetic_masks_dir):
            self.log_output.append("<span style='color: orange;'>⚠️ Invalid masks folder.</span>")
            return
            
        synthetic_tool = os.path.basename(os.path.normpath(synthetic_masks_dir)) or "synthetic_masks"
        
        # Switch to pipeline tab to show progress
        for i in range(self.centralWidget().layout().itemAt(1).widget().count()):
            if "Run Pipeline" in self.centralWidget().layout().itemAt(1).widget().tabText(i):
                self.centralWidget().layout().itemAt(1).widget().setCurrentIndex(i)
                break
                
        self.log_output.clear()
        self.log_output.append(f"<span style='color: #4CAF50;'>🚀 Starting synthetic dashboard pipeline for folder: {synthetic_masks_dir}</span>")
        
        self._synthetic_mode_active = True
        self._synthetic_masks_dir = os.path.abspath(synthetic_masks_dir)
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.current_step = 0
        self.current_tool_index = 0
        self.tools_to_process = [synthetic_tool]
        self.steps_to_run = [
            ("Step 3: Analyze and Plot", step3_analyze_and_plot.run),
            ("Step 4: Process and Plot", step4_process_and_plot.run)
        ]
        
        self.config = self._build_config_for_synthetic_masks(self._synthetic_masks_dir, synthetic_tool)
        
        # Override BLURRED_DIR if realistic dir is provided
        real_dir = self.synthetic_real_dir_input.text().strip()
        if real_dir and os.path.isdir(real_dir):
            self.config['BLURRED_DIR'] = real_dir
        else:
            self.config['BLURRED_DIR'] = None
            
        # Override CSV if provided
        custom_csv = self.synthetic_csv_input.text().strip()
        if custom_csv and os.path.isfile(custom_csv):
            self.config['PROCESSED_CSV_PATH'] = custom_csv
            self.config['ROI_CSV_PATH'] = custom_csv # Fallback
            
        # Check if CSV already exists
        csv_path = self.config['PROCESSED_CSV_PATH']
        if not os.path.exists(csv_path):
            csv_path = self.config['ROI_CSV_PATH']
            
        if os.path.exists(csv_path):
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, 'Analysis Exists',
                f"Data already exists at:\n{csv_path}\n\nDo you want to re-run the analysis, or skip directly to generating the dashboard?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            # Yes = re-run, No = skip to dashboard
            if reply == QMessageBox.StandardButton.No:
                self.log_output.append("<span style='color: #2196F3;'>🌐 Skipping analysis, generating HTML Dashboard...</span>")
                success = dashboard_generator.generate_dashboard(
                    csv_path, 
                    self.config['FINAL_MASKS_DIR'], 
                    self.config['ANALYSIS_OUTPUT_DIR'],
                    real_dir=self.config.get('BLURRED_DIR'),
                    output_filename="dashboard.html",
                    auto_open=True
                )
                if success:
                    self.log_output.append("<span style='color: #4CAF50;'>✓ Dashboard opened in browser!</span>")
                else:
                    self.log_output.append("<span style='color: #f44336;'>✗ Failed to generate dashboard.</span>")
                
                self.run_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
        
        # Add flag to know we should generate dashboard after
        self._generate_dashboard_after = True
        
        self._run_next_step()
        
    def _open_existing_dashboard(self):
        """Open dashboard using specific paths."""
        masks_dir = self.synthetic_masks_dir_input.text().strip()
        if not masks_dir or not os.path.isdir(masks_dir):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Not Found", "Please specify a valid Masks folder.")
            return
            
        real_dir = self.synthetic_real_dir_input.text().strip()
        if not real_dir or not os.path.isdir(real_dir):
            real_dir = None
            
        custom_csv = self.synthetic_csv_input.text().strip()
        
        if custom_csv and os.path.isfile(custom_csv):
            csv_path = custom_csv
            analysis_dir = os.path.dirname(custom_csv)
        else:
            analysis_dir = os.path.join(masks_dir, "analysis")
            csv_path = os.path.join(analysis_dir, "synthetic_masks_processed_data.csv")
            if not os.path.exists(csv_path):
                # Try raw
                csv_path = os.path.join(analysis_dir, "synthetic_masks_raw_data.csv")
                
        if not os.path.exists(csv_path):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Not Found", f"Could not find CSV at {csv_path}.\nPlease select a valid CSV file.")
            return
            
        # Generate on the fly
        self.log_output.append("<span style='color: #2196F3;'>🌐 Generating HTML Dashboard...</span>")
        success = dashboard_generator.generate_dashboard(
            csv_path, 
            masks_dir, 
            analysis_dir,
            real_dir=real_dir,
            output_filename="dashboard.html",
            auto_open=True
        )
        if success:
            self.log_output.append("<span style='color: #4CAF50;'>✓ Dashboard opened in browser!</span>")
        else:
            self.log_output.append("<span style='color: #f44336;'>✗ Failed to generate dashboard.</span>")

    def _browse_compare_A(self):
        start_dir = self.compare_A_input.text().strip() or self.SYNTHETIC_DEFAULT_ROOT
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Tool A Masks Folder", start_dir)
        if selected_dir:
            self.compare_A_input.setText(selected_dir)
            
    def _browse_compare_B(self):
        start_dir = self.compare_B_input.text().strip() or self.SYNTHETIC_DEFAULT_ROOT
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Tool B Masks Folder", start_dir)
        if selected_dir:
            self.compare_B_input.setText(selected_dir)
            
    def _run_compare_dashboard(self):
        dir_A = self.compare_A_input.text().strip()
        dir_B = self.compare_B_input.text().strip()
        
        if not dir_A or not os.path.isdir(dir_A):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Invalid directory for Tool A")
            return
        if not dir_B or not os.path.isdir(dir_B):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Invalid directory for Tool B")
            return
            
        import image_to_signal.compare_dashboard_generator as comp_gen
        
        # Output in the parent directory of Tool A's analysis
        analysis_dir = os.path.join(dir_A, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        self.log_output.append("<span style='color: #2196F3;'>⚖️ Generating Comparison Dashboard...</span>")
        success = comp_gen.generate_comparison_dashboard(dir_A, dir_B, analysis_dir)
        if success:
            self.log_output.append("<span style='color: #4CAF50;'>✓ Comparison Dashboard generated!</span>")
        else:
            self.log_output.append("<span style='color: #f44336;'>✗ Failed to generate comparison dashboard. Check if tools have been analyzed first.</span>")

    def _run_next_step(self):
        """Run the next step in the queue."""
        # Check if we finished all steps for current tool
        if self.current_step >= len(self.steps_to_run):
            # Move to next tool
            self.current_tool_index += 1
            if self.current_tool_index >= len(self.tools_to_process):
                # All tools processed
                self._pipeline_finished()
                return
            
            # Start next tool
            self.current_step = 0
            current_tool = self.tools_to_process[self.current_tool_index]
            self.log_output.append(f"\n<span style='color: #FFA726;'>📦 Processing tool {self.current_tool_index + 1}/{len(self.tools_to_process)}: {current_tool}</span>")
            # Update config with new tool_id
            self.config['tool_id'] = current_tool
            if self._synthetic_mode_active:
                self.config = self._build_config_for_synthetic_masks(
                    self._synthetic_masks_dir, current_tool
                )
            else:
                self.config = self._build_config_for_tool(current_tool)
        
        step_name, step_func = self.steps_to_run[self.current_step]
        current_tool = self.tools_to_process[self.current_tool_index]
        self.log_output.append(f"\n<span style='color: #2196F3;'>▶️ {step_name} for {current_tool}...</span>")
        self.log_output.append(f"<span style='color: #888;'>(Check terminal for detailed progress)</span>")
        self.status_label.setText(f"Running: {step_name} ({current_tool})")
        
        # Run in thread
        self.worker = ProcessingThread(step_func, self.config)
        self.worker.finished.connect(self._step_finished)
        self.worker.error.connect(self._step_error)
        self.worker.start()
    
    def _step_finished(self, message):
        """Handle step completion."""
        self.log_output.append(f"<span style='color: #4CAF50;'>✓ {message}</span>")
        self.current_step += 1
        QTimer.singleShot(100, self._run_next_step)
    
    def _step_error(self, message):
        """Handle step error."""
        self.log_output.append(f"<span style='color: #f44336;'>✗ {message}</span>")
        self.log_output.append(f"<span style='color: #FFA726;'>⚠️ Skipping remaining steps for this tool due to error.</span>")
        # Skip to next tool
        self.current_step = len(self.steps_to_run)
        QTimer.singleShot(100, self._run_next_step)
    
    def _stop_pipeline(self):
        """Stop the running pipeline."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.log_output.append(f"\n<span style='color: #FFA726;'>⏹️ Stopping pipeline...</span>")
            self.worker.terminate()
            self.worker.wait()
            self.log_output.append(f"<span style='color: #FFA726;'>⚠️ Pipeline stopped by user.</span>")
        
        self.status_label.setText("Stopped")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _pipeline_finished(self):
        """Handle pipeline completion."""
        self.log_output.append(f"\n<span style='color: #4CAF50;'>✓ Pipeline finished! Processed {len(self.tools_to_process)} tool(s).</span>")
        self.status_label.setText("Ready")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Generate Dashboard if flag is set (Synthetic mode)
        if getattr(self, '_generate_dashboard_after', False):
            self.log_output.append(f"<span style='color: #2196F3;'>🌐 Generating HTML Dashboard...</span>")
            
            # Prefer processed csv, fallback to raw csv
            csv_path = self.config['PROCESSED_CSV_PATH']
            if not os.path.exists(csv_path):
                csv_path = self.config['ROI_CSV_PATH']
                
            if os.path.exists(csv_path):
                success = dashboard_generator.generate_dashboard(
                    csv_path, 
                    self.config['FINAL_MASKS_DIR'], 
                    self.config['ANALYSIS_OUTPUT_DIR'],
                    real_dir=self.config.get('BLURRED_DIR'),
                    output_filename="dashboard.html",
                    auto_open=True
                )
                if success:
                    self.log_output.append(f"<span style='color: #4CAF50;'>✓ Dashboard opened in browser!</span>")
                else:
                    self.log_output.append(f"<span style='color: #f44336;'>✗ Failed to generate dashboard.</span>")
            else:
                self.log_output.append(f"<span style='color: #f44336;'>✗ Cannot generate dashboard, CSV not found.</span>")
            
            self._generate_dashboard_after = False

    def _run_find360(self):
        """Run find360; suppress external plot, embed via JSON afterward."""
        self.utils_log_output.append(f"<span style='color:#4CAF50;'>Starting find360 for {self.tool_id} (embedded plot)...</span>")
        self._pending_find360_json = os.path.join(self.DATA_ROOT, '1d_profiles', f'find360_{self.tool_id}.json')
        try:
            if os.path.isfile(self._pending_find360_json):
                os.remove(self._pending_find360_json)
        except Exception:
            pass
        # Temporarily enable JSON for GUI embedding; find360 won't save JSON by default anymore
        self._start_utility_process([sys.executable, "-u", "-m", "image_to_signal.find360", "--tool", self.tool_id, "--no-plot", "--write-json"]) 

    def _remove_extra_images(self):
        """Remove extra images from both masks and blurred folders."""
        from PyQt6.QtWidgets import QMessageBox
        
        result = getattr(self, '_last_find360_result', None)
        if not result:
            QMessageBox.warning(self, "No Detection Data", "Please run the similarity detector first.")
            return
        
        best_frame = result.get('best_frame_number')
        if not best_frame:
            QMessageBox.warning(self, "Invalid Data", "Detection result doesn't contain frame count.")
            return
        
        tool_id = result.get('tool_id', self.tool_id)
        masks_dir = os.path.join(self.DATA_ROOT, 'masks', f"{tool_id}_final_masks")
        blurred_dir = os.path.join(self.DATA_ROOT, 'blurred', f"{tool_id}_blurred")
        
        # Count files to remove
        mask_files = []
        blurred_files = []
        if os.path.isdir(masks_dir):
            mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
        if os.path.isdir(blurred_dir):
            blurred_files = sorted([f for f in os.listdir(blurred_dir) if f.lower().endswith(('.tiff', '.tif'))])
        
        mask_extra = len(mask_files) - best_frame if len(mask_files) > best_frame else 0
        blurred_extra = len(blurred_files) - best_frame if len(blurred_files) > best_frame else 0
        
        if mask_extra == 0 and blurred_extra == 0:
            QMessageBox.information(self, "No Extra Images", "No extra images found to remove.")
            return
        
        # Confirm deletion
        msg = f"This will permanently delete:\n\n"
        if mask_extra > 0:
            msg += f"• {mask_extra} image(s) from masks folder\n"
        if blurred_extra > 0:
            msg += f"• {blurred_extra} image(s) from blurred folder\n"
        msg += f"\n(Removing last {max(mask_extra, blurred_extra)} files from each folder)\n\nContinue?"
        
        reply = QMessageBox.question(
            self, 
            'Confirm Deletion',
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            self.utils_log_output.append("<span style='color:#FFA726;'>Deletion cancelled by user.</span>")
            return
        
        # Delete extra files
        deleted_count = 0
        try:
            if mask_extra > 0:
                for f in mask_files[-mask_extra:]:
                    os.remove(os.path.join(masks_dir, f))
                    deleted_count += 1
                self.utils_log_output.append(f"<span style='color:#4CAF50;'>✓ Deleted {mask_extra} file(s) from masks folder</span>")
            
            if blurred_extra > 0:
                for f in blurred_files[-blurred_extra:]:
                    os.remove(os.path.join(blurred_dir, f))
                    deleted_count += 1
                self.utils_log_output.append(f"<span style='color:#4CAF50;'>✓ Deleted {blurred_extra} file(s) from blurred folder</span>")
            
            self.utils_log_output.append(f"<span style='color:#4CAF50;'>✓ Total: {deleted_count} file(s) deleted successfully</span>")
            
            # Update button state
            self.remove_extra_btn.setEnabled(False)
            self.remove_extra_label.setText("Extra images removed. Image count now matches 360° frame count.")
            self.remove_extra_label.setStyleSheet("color: #4CAF50; font-style: italic;")
            
            QMessageBox.information(self, "Success", f"Successfully deleted {deleted_count} file(s).")
            
        except Exception as e:
            self.utils_log_output.append(f"<span style='color:#f44336;'>✗ Error during deletion: {e}</span>")
            QMessageBox.critical(self, "Error", f"Failed to delete files:\n{str(e)}")
    
    def _run_rename_all(self):
        """Run rename_by_angle for both masks and blurred folders."""
        frames360 = self.frames360_spin.value()
        
        # Check if already renamed
        masks_dir = os.path.join(self.DATA_ROOT, "masks", f"{self.tool_id}_final_masks")
        blurred_dir = os.path.join(self.DATA_ROOT, "blurred", f"{self.tool_id}_blurred")
        
        masks_already_renamed = False
        blurred_already_renamed = False
        
        if os.path.isdir(masks_dir):
            files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
            masks_already_renamed = any('_degrees.png' in f for f in files)
        
        if os.path.isdir(blurred_dir):
            files = [f for f in os.listdir(blurred_dir) if f.lower().endswith(('.tiff', '.tif'))]
            blurred_already_renamed = any('_degrees.tiff' in f for f in files)
        
        if masks_already_renamed or blurred_already_renamed:
            from PyQt6.QtWidgets import QMessageBox
            msg = "Images appear to be already renamed with angle values:\n\n"
            if masks_already_renamed:
                msg += "• Masks folder already renamed\n"
            if blurred_already_renamed:
                msg += "• Blurred folder already renamed\n"
            msg += "\nDo you want to rename them again (override)?"
            
            reply = QMessageBox.question(
                self, 
                'Files Already Renamed',
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.utils_log_output.append("<span style='color:#FFA726;'>Rename cancelled by user.</span>")
                return
            force_flag = ["--force"]
        else:
            force_flag = []
        
        # Store state for sequential execution
        self._rename_queue = ['masks', 'blurred']
        self._rename_frames360 = frames360
        self._rename_force_flag = force_flag
        
        self.utils_log_output.append(f"<span style='color:#4CAF50;'>Renaming all images for {self.tool_id} with frames360={frames360}...</span>")
        self._run_next_rename()
    
    def _run_next_rename(self):
        """Run the next folder rename in the queue."""
        if not self._rename_queue:
            self.utils_log_output.append(f"<span style='color:#4CAF50;'>✓ All folders renamed successfully!</span>")
            return
        
        folder = self._rename_queue.pop(0)
        self.utils_log_output.append(f"<span style='color:#2196F3;'>→ Renaming {folder} folder...</span>")
        
        # Start the rename process for this folder
        cmd = [sys.executable, "-u", "-m", "image_to_signal.rename_by_angle", 
               "--tool", self.tool_id, "--frames360", str(self._rename_frames360),
               "--folder", folder] + self._rename_force_flag
        self._start_utility_process(cmd, on_complete=self._run_next_rename)

    def _start_utility_process(self, cmd, on_complete=None):
        """Start utility subprocess and stream output to utils log."""
        try:
            import subprocess
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self._proc_on_complete = on_complete
            self._poll_process_output()
        except Exception as e:
            self.utils_log_output.append(f"<span style='color:#f44336;'>Error starting process: {e}</span>")

    def _poll_process_output(self):
        """Poll running process output (non-blocking using QTimer)."""
        if not hasattr(self, 'proc'):
            return
        # Read any available lines first
        while True:
            if self.proc.stdout is None:
                break
            pos = self.proc.stdout.tell()
            line = self.proc.stdout.readline()
            if not line:
                # restore position if no full line
                try:
                    self.proc.stdout.seek(pos)
                except Exception:
                    pass
                break
            safe = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
            self.utils_log_output.append(safe)
        # If finished, drain remaining buffer then show code
        if self.proc.poll() is not None:
            # Drain any residual text
            residual = self.proc.stdout.read()
            if residual:
                for line in residual.splitlines():
                    safe = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
                    self.utils_log_output.append(safe)
            rc = self.proc.returncode
            color = '#4CAF50' if rc == 0 else '#f44336'
            self.utils_log_output.append(f"<span style='color:{color};'>Process finished (code {rc}).</span>")
            
            # Handle completion callback
            if rc == 0 and hasattr(self, '_pending_find360_json'):
                # Defer loading slightly in case file buffer not flushed yet
                QTimer.singleShot(120, self._load_find360_json_and_plot)
            elif hasattr(self, '_proc_on_complete') and self._proc_on_complete:
                # Execute the completion callback
                callback = self._proc_on_complete
                self._proc_on_complete = None
                QTimer.singleShot(100, callback)
            return
        line = self.proc.stdout.readline()
        if line:
            safe = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
            self.utils_log_output.append(safe)
        QTimer.singleShot(50, self._poll_process_output)

    def _load_find360_json_and_plot(self):
        """Load JSON results from find360 and render embedded plot."""
        path = getattr(self, '_pending_find360_json', None)
        if not path:
            return
        if not os.path.isfile(path):
            self.utils_log_output.append("<span style='color:#FFA726;'>JSON not found for plotting.</span>")
            return
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.utils_log_output.append(f"<span style='color:#f44336;'>Failed JSON read: {e}</span>")
            return

        # Store detection results for later use
        self._last_find360_result = data
        best_frame = data.get('best_frame_number', None)
        
        # Update frames360 spinbox with detected value
        if best_frame is not None:
            self.frames360_spin.setValue(best_frame)
            self.utils_log_output.append(f"<span style='color:#4CAF50;'>✓ Auto-set frames for 360° to {best_frame}</span>")
        
        # Update remove button based on extra images
        masks_dir = os.path.join(self.DATA_ROOT, 'masks', f"{data.get('tool_id','')}_final_masks")
        blurred_dir = os.path.join(self.DATA_ROOT, 'blurred', f"{data.get('tool_id','')}_blurred")
        
        extra_count = 0
        if os.path.isdir(masks_dir):
            mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
            if best_frame and len(mask_files) > best_frame:
                extra_count = len(mask_files) - best_frame
        
        if extra_count > 0:
            self.remove_extra_label.setText(f"Detected {extra_count} extra image(s) beyond 360° cycle.")
            self.remove_extra_label.setStyleSheet("color: #FFA726; font-weight: bold;")
            self.remove_extra_btn.setText(f"🗑️ Remove {extra_count} Extra Image(s)")
            self.remove_extra_btn.setEnabled(True)
        else:
            self.remove_extra_label.setText("No extra images detected. Image count matches 360° frame count.")
            self.remove_extra_label.setStyleSheet("color: #4CAF50; font-style: italic;")
            self.remove_extra_btn.setEnabled(False)

        # Clear and redraw figure
        self.find360_fig.clear()
        ax_main = self.find360_fig.add_subplot(2, 2, (1, 3))
        ax_ref = self.find360_fig.add_subplot(2, 2, 2)
        ax_best = self.find360_fig.add_subplot(2, 2, 4)

        nums = data.get('frame_numbers', [])
        white_i = data.get('white_i_series', [])
        white_j = data.get('white_j_series', [])
        first_white = data.get('first_white', None)
        second_white = data.get('second_white', None)

        if nums and white_i:
            ax_main.plot(nums, white_i, 'b-', label='Frame i', linewidth=1.1)
        if nums and white_j:
            ax_main.plot([n + 1 for n in nums], white_j, 'c--', label='Frame i+1', linewidth=1.0)
        if first_white is not None:
            ax_main.axhline(first_white, color='green', linestyle='--', linewidth=0.8, label=f'Ref1 {first_white:,}')
        if second_white is not None:
            ax_main.axhline(second_white, color='lime', linestyle='--', linewidth=0.8, label=f'Ref2 {second_white:,}')
        if best_frame is not None:
            ax_main.axvline(best_frame, color='red', linewidth=1.2, label=f'Best {best_frame}')
        ax_main.set_xlabel('Frame Number')
        ax_main.set_ylabel('White Pixels')
        ax_main.set_title(f"360° Similarity (tool {data.get('tool_id','?')})")
        ax_main.legend(fontsize=7, ncol=2)
        ax_main.grid(alpha=0.25)

        # Attempt side images
        try:
            mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
            if len(mask_files) >= 2:
                from PIL import Image
                import numpy as np
                ref1 = np.array(Image.open(os.path.join(masks_dir, mask_files[0])))
                ref2 = np.array(Image.open(os.path.join(masks_dir, mask_files[1])))
                ax_ref.imshow(np.hstack([ref1, ref2]), cmap='gray')
                ax_ref.axis('off')
                ax_ref.set_title(f"Ref 1 & 2\n{first_white:,} | {second_white:,}", fontsize=8)
            if best_frame is not None and best_frame - 1 < len(mask_files) - 1:
                bf_idx = best_frame - 1
                best_i = np.array(Image.open(os.path.join(masks_dir, mask_files[bf_idx])))
                best_j = np.array(Image.open(os.path.join(masks_dir, mask_files[bf_idx + 1])))
                ax_best.imshow(np.hstack([best_i, best_j]), cmap='gray')
                ax_best.axis('off')
                ax_best.set_title(f"Best {best_frame} & {best_frame+1}\n{data.get('best_white_i','?')} | {data.get('best_white_j','?')}", fontsize=8)
        except Exception as e:
            ax_ref.set_title('Ref load error')
            ax_best.set_title('Best load error')
            self.utils_log_output.append(f"<span style='color:#FFA726;'>Image load warning: {e}</span>")

        self.find360_fig.tight_layout()
        self.find360_canvas.draw()
        self.utils_log_output.append("<span style='color:#4CAF50;'>Embedded plot updated.</span>")
        
        # Clean up JSON file after loading
        try:
            if os.path.isfile(path):
                os.remove(path)
            del self._pending_find360_json
        except Exception:
            pass
    
    def _open_file(self, config_key):
        """Open a file or folder in the default application."""
        path = self.config.get(config_key)
        if not path:
            self.log_output.append(f"<span style='color: orange;'>⚠️ Path not configured for {config_key}</span>")
            return
        
        if not os.path.exists(path):
            self.log_output.append(f"<span style='color: orange;'>⚠️ File/folder not found: {path}</span>")
            return
        
        try:
            import subprocess
            if os.path.isfile(path):
                # Open file with default application
                os.startfile(path)
                self.log_output.append(f"<span style='color: #4CAF50;'>✓ Opened: {os.path.basename(path)}</span>")
            else:
                # Open folder in explorer
                subprocess.Popen(f'explorer "{path}"')
                self.log_output.append(f"<span style='color: #4CAF50;'>✓ Opened folder: {os.path.basename(path)}</span>")
        except Exception as e:
            self.log_output.append(f"<span style='color: #f44336;'>✗ Error opening: {str(e)}</span>")
    
    def _apply_styles(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #4CAF50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #ddd;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #1e1e1e;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: #fff;
            }
            QCheckBox {
                spacing: 5px;
                color: #ddd;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #555;
            }
            QCheckBox::indicator:checked {
                background: #4CAF50;
                border-color: #4CAF50;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background: #2d2d2d;
            }
            QTabBar::tab {
                background: #1e1e1e;
                color: #888;
                padding: 10px 20px;
                border: 1px solid #444;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #2d2d2d;
                color: #4CAF50;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background: #1e1e1e;
            }
            QProgressBar::chunk {
                background: #4CAF50;
            }
            QScrollArea {
                border: none;
            }
            QFrame {
                border-radius: 5px;
                background: #1e1e1e;
                padding: 10px;
            }
        """)
    
    def _open_tool_selector(self):
        """Open dialog for multi-selecting tool IDs."""
        # Read available tool IDs from metadata CSV
        import csv
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "DATA")
        metadata_path = os.path.join(data_root, "tools_metadata.csv")
        
        available_tools = []
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tool_id = row.get('tool_id', '')
                        if tool_id:
                            available_tools.append(tool_id)
            except Exception as e:
                print(f"Error reading metadata: {e}")
        
        if not available_tools:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Tools Found", "Could not find any tools in tools_metadata.csv")
            return
        
        # Open selector dialog
        dialog = ToolSelectorDialog(available_tools, self.tool_id_input.text(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = dialog.get_selected_tools()
            if selected:
                self.tool_id_input.setText(", ".join(selected))


class ToolSelectorDialog(QDialog):
    """Dialog for selecting multiple tool IDs with checkboxes."""
    
    def __init__(self, available_tools, current_selection, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tool IDs")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
        
        # Apply dark theme styling with green checkmarks
        self.setStyleSheet("""
            QListWidget {
                background: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #555;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3d3d3d;
            }
            QListWidget::item:hover {
                background: #3d3d3d;
            }
            QListWidget::item:selected {
                background: #4d4d4d;
            }
            QListWidget::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555;
                border-radius: 3px;
                background: #1e1e1e;
            }
            QListWidget::indicator:checked {
                background: #4CAF50;
                border-color: #4CAF50;
                image: url(none);
            }
            QListWidget::indicator:checked::after {
                content: "✓";
                color: white;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Info label
        info = QLabel("Select tools to process (Shift for range, Ctrl for individual):")
        layout.addWidget(info)
        
        # Select/Deselect All buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(deselect_all_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # List widget with checkboxes
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        
        # Parse current selection
        current_tools = [t.strip() for t in current_selection.split(',') if t.strip()]
        
        # Add tools as checkable items
        for tool_id in sorted(available_tools):
            item = QListWidgetItem(tool_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if tool_id in current_tools else Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
        
        layout.addWidget(self.list_widget)
        
        # OK/Cancel buttons
        btn_box_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box_layout.addStretch()
        btn_box_layout.addWidget(ok_btn)
        btn_box_layout.addWidget(cancel_btn)
        layout.addLayout(btn_box_layout)
    
    def _select_all(self):
        """Check all items."""
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)
    
    def _deselect_all(self):
        """Uncheck all items."""
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)
    
    def get_selected_tools(self):
        """Return list of checked tool IDs."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected


def main():
    # Windows-specific: Set AppUserModelID for proper taskbar icon grouping
    try:
        import ctypes
        myappid = 'alireza.toolconditionmonitoring.imageprocessing.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass
    
    app = QApplication(sys.argv)
    # Set application icon (for taskbar)
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    icon_path = os.path.join(SCRIPT_DIR, "app_icon.ico")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = ImageToSignalGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
