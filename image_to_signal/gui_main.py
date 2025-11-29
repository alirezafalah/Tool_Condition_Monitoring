"""
Sophisticated GUI for Image-to-Signal Processing Pipeline
"""
import sys
import os
import warnings
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QGroupBox, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QComboBox, QProgressBar, QTextEdit, QTabWidget,
                            QFileDialog, QFrame, QScrollArea, QDialog, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

# Suppress matplotlib threading warnings
warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside of the main thread')

# Import processing modules
from . import step1_blur_and_rename
from . import step2_generate_masks
from . import step3_analyze_and_plot
from . import step4_process_and_plot


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


class ImageToSignalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image-to-Signal Processing Pipeline")
        self.setGeometry(100, 100, 1200, 800)
        
        # Calculate DATA_ROOT
        SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATA"))
        
        # Initialize config
        self.tool_id = 'tool'  # Start with 'tool' prefix
        self.config = self._create_default_config()
        
        # Setup UI
        self._setup_ui()
        self._apply_styles()
        
    def _create_default_config(self):
        """Create default configuration dictionary."""
        return {
            'RAW_DIR': os.path.join(self.DATA_ROOT, 'tools', self.tool_id),
            'BLURRED_DIR': os.path.join(self.DATA_ROOT, 'blurred', f'{self.tool_id}_blurred'),
            'FINAL_MASKS_DIR': os.path.join(self.DATA_ROOT, 'masks', f'{self.tool_id}_final_masks'),
            'ROI_CSV_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_area_vs_angle.csv'),
            'ROI_PLOT_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_area_vs_angle_plot.svg'),
            'PROCESSED_CSV_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_area_vs_angle_processed.csv'),
            'PROCESSED_PLOT_PATH': os.path.join(self.DATA_ROOT, '1d_profiles', f'{self.tool_id}_area_vs_angle_processed_plot.svg'),
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
            'WHITE_RATIO_OUTLIER_THRESHOLD': 0.8,
            'APPLY_MOVING_AVERAGE': True,
            'MOVING_AVERAGE_WINDOW': 5,
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
        main_layout.addWidget(tabs, stretch=1)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background: #2d2d2d; color: #4CAF50; border-radius: 3px;")
        main_layout.addWidget(self.status_label)
    
    def _create_header(self):
        """Create header with title and tool selector."""
        header = QFrame()
        header.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(header)
        
        title = QLabel("🔧 Image-to-Signal Processing")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #4CAF50;")
        
        layout.addWidget(title)
        layout.addStretch()
        
        # Tool ID selector
        layout.addWidget(QLabel("Tool ID:"))
        self.tool_id_input = QLineEdit(self.tool_id)
        self.tool_id_input.setMaximumWidth(200)
        self.tool_id_input.textChanged.connect(self._on_tool_id_changed)
        layout.addWidget(self.tool_id_input)
        
        # Multi-select button
        multi_select_btn = QPushButton("📋 Select Multiple")
        multi_select_btn.setMaximumWidth(120)
        multi_select_btn.clicked.connect(self._open_tool_selector)
        layout.addWidget(multi_select_btn)
        
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
        
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.bg_method = QComboBox()
        self.bg_method.addItems(['none', 'absdiff', 'lab'])
        self.bg_method.setCurrentText(self.config['BACKGROUND_SUBTRACTION_METHOD'])
        method_row.addWidget(self.bg_method)
        method_row.addStretch()
        bg_layout.addLayout(method_row)
        
        self.apply_multichannel = QCheckBox("Apply Multichannel Mask")
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
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        info = QLabel("Tools to detect 360° frame count and rename raw images based on that count.")
        info.setStyleSheet("color: #ccc; font-style: italic;")
        layout.addWidget(info)

        # Two groups side by side - equal width
        top_row = QHBoxLayout()
        
        # Detection group
        detect_group = QGroupBox("Find 360° Frame Count")
        dg_layout = QVBoxLayout()
        find_btn = QPushButton("🔍 Run Similarity Detector\n(find360)")
        find_btn.setMinimumHeight(60)
        find_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        find_btn.clicked.connect(self._run_find360)
        dg_layout.addWidget(find_btn)
        dg_layout.addStretch()
        detect_group.setLayout(dg_layout)
        top_row.addWidget(detect_group, 1)

        # Rename group
        rename_group = QGroupBox("Rename Raw Images")
        rg_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Frames for 360°:"))
        self.frames360_spin = QSpinBox(); self.frames360_spin.setRange(1, 5000); self.frames360_spin.setValue(360)
        self.frames360_spin.setMinimumWidth(100)
        row.addWidget(self.frames360_spin)
        row.addStretch()
        rg_layout.addLayout(row)
        rename_btn = QPushButton("📝 Rename Using Above Value")
        rename_btn.setMinimumHeight(40)
        rename_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        rename_btn.clicked.connect(self._run_rename)
        rg_layout.addWidget(rename_btn)
        rename_group.setLayout(rg_layout)
        top_row.addWidget(rename_group, 1)
        
        layout.addLayout(top_row)

        # Live output
        log_group = QGroupBox("360° Utilities Output")
        lg_layout = QVBoxLayout()
        self.utils_log_output = QTextEdit()
        self.utils_log_output.setReadOnly(True)
        self.utils_log_output.setStyleSheet("background:#1e1e1e; color:#ccc; font-family:Consolas;")
        lg_layout.addWidget(self.utils_log_output)
        log_group.setLayout(lg_layout)
        layout.addWidget(log_group)

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
    
    def _on_tool_id_changed(self):
        """Update paths when tool ID changes."""
        self.tool_id = self.tool_id_input.text()
        self.config = self._create_default_config()
        
        # Update path labels
        for key, label in self.path_labels.items():
            label.setText(self.config[key])
    
    def _update_config_from_ui(self):
        """Update config dictionary from UI values."""
        self.config['blur_kernel'] = self.blur_kernel.value()
        self.config['closing_kernel'] = self.closing_kernel.value()
        self.config['BACKGROUND_SUBTRACTION_METHOD'] = self.bg_method.currentText()
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
        config['AREA_VS_ANGLE_CSV'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_area_vs_angle.csv')
        config['PROCESSED_CSV_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_area_vs_angle_processed.csv')
        config['ROI_PLOT_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_roi_plot.svg')
        config['PROCESSED_PLOT_PATH'] = os.path.join(data_root, '1d_profiles', f'{tool_id}_processed_plot.svg')
        
        return config
    
    def _run_pipeline(self):
        """Run selected pipeline steps."""
        self._update_config_from_ui()
        
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
    
    def _pipeline_finished(self):
        """Handle pipeline completion."""
        self.log_output.append(f"\n<span style='color: #4CAF50;'>✓ Pipeline finished! Processed {len(self.tools_to_process)} tool(s).</span>")
        self.status_label.setText("Ready")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _run_find360(self):
        """Run find360 with current tool id, show plot window, capture output."""
        self.utils_log_output.append(f"<span style='color:#4CAF50;'>Starting find360 for {self.tool_id}...</span>")
        self._start_utility_process([sys.executable, "-u", "-m", "image_to_signal.find360", "--tool", self.tool_id])

    def _run_rename(self):
        """Run rename_by_angle capturing output, with confirmation if already renamed."""
        frames360 = self.frames360_spin.value()
        self.utils_log_output.append(f"<span style='color:#4CAF50;'>Checking mask files for {self.tool_id}...</span>")
        
        # First check if already renamed
        import os
        # DATA is at CCD_DATA/DATA level (sibling to Tool_Condition_Monitoring)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Tool_Condition_Monitoring
        data_root = os.path.join(os.path.dirname(script_dir), "DATA")  # Go up one more level to CCD_DATA, then DATA
        masks_dir = os.path.join(data_root, "masks", f"{self.tool_id}_final_masks")
        
        if os.path.isdir(masks_dir):
            files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.tiff', '.tif'))]
            already_renamed = any('_degrees.tiff' in f for f in files)
            
            if already_renamed:
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, 
                    'Files Already Renamed',
                    f'Mask files in {self.tool_id}_final_masks appear to be already renamed with angle values.\n\nDo you want to rename them again (override)?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    self.utils_log_output.append("<span style='color:#FFA726;'>Rename cancelled by user.</span>")
                    return
                # User said yes, add --force flag
                self.utils_log_output.append(f"<span style='color:#4CAF50;'>Re-renaming mask files for {self.tool_id} with frames360={frames360}...</span>")
                self._start_utility_process([sys.executable, "-u", "-m", "image_to_signal.rename_by_angle", "--tool", self.tool_id, "--frames360", str(frames360), "--force"])
                return
        
        # Not renamed yet, proceed normally
        self.utils_log_output.append(f"<span style='color:#4CAF50;'>Renaming mask files for {self.tool_id} with frames360={frames360}...</span>")
        self._start_utility_process([sys.executable, "-u", "-m", "image_to_signal.rename_by_angle", "--tool", self.tool_id, "--frames360", str(frames360)])

    def _start_utility_process(self, cmd):
        """Start utility subprocess and stream output to utils log."""
        try:
            import subprocess
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            return
        line = self.proc.stdout.readline()
        if line:
            safe = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
            self.utils_log_output.append(safe)
        QTimer.singleShot(50, self._poll_process_output)
    
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
    app = QApplication(sys.argv)
    window = ImageToSignalGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
