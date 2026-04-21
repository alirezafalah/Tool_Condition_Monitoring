import os
import sys
import glob
import re
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QSpinBox, 
    QCheckBox, QGroupBox, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def extract_frame_num(filepath):
    """Extract frame number or angle from filename."""
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    # First try beginning of file
    match = re.match(r"^(\d+\.?\d*)", name)
    if match:
        return float(match.group(1))
    # Then try finding numbers separated by underscores
    parts = name.split("_")
    for part in reversed(parts):
        try:
            return float(part)
        except ValueError:
            continue
    return 0.0


class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, int) # df, master_mask, centerline
    error = pyqtSignal(str)

    def __init__(self, masks_dir, roi_height, full_picture):
        super().__init__()
        self.masks_dir = masks_dir
        self.roi_height = roi_height
        self.full_picture = full_picture

    def run(self):
        try:
            # 1. Find all images
            self.progress.emit(f"Scanning directory: {self.masks_dir}")
            image_files = []
            for ext in ('*.png', '*.tiff', '*.tif', '*.jpg', '*.jpeg'):
                image_files.extend(glob.glob(os.path.join(self.masks_dir, ext)))
            
            if not image_files:
                self.error.emit("No images found in the selected directory.")
                return

            image_files.sort(key=extract_frame_num)
            self.progress.emit(f"Found {len(image_files)} images. Building master mask...")

            # 2. Build Master Mask
            first_img = np.array(Image.open(image_files[0]))
            master_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)

            for i, fpath in enumerate(image_files):
                if i % 10 == 0:
                    self.progress.emit(f"Building master mask: {i}/{len(image_files)}")
                img = np.array(Image.open(fpath))
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                master_mask = cv2.bitwise_or(master_mask, binary)

            self.progress.emit("Master mask built. Finding centerline...")

            # 3. Find Centerline
            white_coords = np.where(master_mask == 255)
            if white_coords[1].size == 0:
                self.error.emit("Master mask is completely black. Cannot find centerline.")
                return
            
            min_x = white_coords[1].min()
            max_x = white_coords[1].max()
            centerline = int((min_x + max_x) / 2)
            self.progress.emit(f"Centerline found at X={centerline} (Tool width: {min_x} to {max_x})")

            # 4. Analyze each frame
            results = []
            for i, fpath in enumerate(image_files):
                if i % 10 == 0:
                    self.progress.emit(f"Analyzing right half: frame {i}/{len(image_files)}")
                
                angle = extract_frame_num(fpath)
                img = np.array(Image.open(fpath))
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                
                # Keep only right half
                right_half = binary[:, centerline:]
                
                white_pixels = np.where(right_half == 255)
                
                if self.full_picture or self.roi_height <= 0:
                    area = white_pixels[0].size
                else:
                    if white_pixels[0].size == 0:
                        area = 0
                    else:
                        last_row = white_pixels[0].max()
                        first_row = max(0, last_row - self.roi_height)
                        roi = right_half[first_row:last_row, :]
                        area = np.sum(roi == 255)
                
                results.append({
                    'Angle': angle,
                    'Area': area,
                    'Filename': os.path.basename(fpath)
                })

            df = pd.DataFrame(results)
            
            # Save CSV
            out_dir = os.path.join(self.masks_dir, "half_tool_analysis")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "right_half_analysis.csv")
            df.to_csv(csv_path, index=False)
            self.progress.emit(f"Analysis saved to {csv_path}")

            self.finished.emit(df, master_mask, centerline)

        except Exception as e:
            self.error.emit(f"Error during analysis: {str(e)}")


class HalfToolAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Half-Tool Symmetry Analysis")
        self.resize(1000, 700)
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. Inputs
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Masks Directory:"))
        self.dir_input = QLineEdit()
        dir_layout.addWidget(self.dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_dir)
        dir_layout.addWidget(browse_btn)
        input_layout.addLayout(dir_layout)

        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI Height:"))
        self.roi_input = QSpinBox()
        self.roi_input.setRange(1, 2000)
        self.roi_input.setValue(200)
        roi_layout.addWidget(self.roi_input)
        
        self.full_pic_checkbox = QCheckBox("Analyze Full Picture (Ignore ROI Height)")
        self.full_pic_checkbox.toggled.connect(lambda checked: self.roi_input.setEnabled(not checked))
        roi_layout.addWidget(self.full_pic_checkbox)
        roi_layout.addStretch()
        input_layout.addLayout(roi_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 2. Run Button
        self.run_btn = QPushButton("🚀 Run Half-Tool Analysis")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 8px;")
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)

        # 3. Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        layout.addWidget(self.log_output)

        # 4. Plots (Split view: Graph + Master Mask)
        plot_layout = QHBoxLayout()
        
        # Line Graph
        self.fig_graph = Figure()
        self.canvas_graph = FigureCanvas(self.fig_graph)
        self.ax_graph = self.fig_graph.add_subplot(111)
        plot_layout.addWidget(self.canvas_graph, stretch=2)

        # Master Mask Preview
        self.fig_mask = Figure()
        self.canvas_mask = FigureCanvas(self.fig_mask)
        self.ax_mask = self.fig_mask.add_subplot(111)
        plot_layout.addWidget(self.canvas_mask, stretch=1)

        layout.addLayout(plot_layout, stretch=1)

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Masks Directory")
        if d:
            self.dir_input.setText(d)

    def log(self, text):
        self.log_output.append(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def run_analysis(self):
        masks_dir = self.dir_input.text().strip()
        if not masks_dir or not os.path.isdir(masks_dir):
            QMessageBox.warning(self, "Error", "Please select a valid masks directory.")
            return

        self.run_btn.setEnabled(False)
        self.log("Starting analysis...")
        
        roi = self.roi_input.value()
        full_pic = self.full_pic_checkbox.isChecked()

        self.worker = AnalysisWorker(masks_dir, roi, full_pic)
        self.worker.progress.connect(self.log)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.handle_finished)
        self.worker.start()

    def handle_error(self, err):
        self.log(f"ERROR: {err}")
        QMessageBox.critical(self, "Error", err)
        self.run_btn.setEnabled(True)

    def handle_finished(self, df, master_mask, centerline):
        self.log("Analysis Complete! Updating plots...")
        
        # Update Graph
        self.ax_graph.clear()
        self.ax_graph.plot(df['Angle'], df['Area'], color='blue', linewidth=2)
        self.ax_graph.set_title("Right-Side ROI Area vs Angle")
        self.ax_graph.set_xlabel("Angle / Frame")
        self.ax_graph.set_ylabel("White Pixel Count")
        self.ax_graph.grid(True)
        self.canvas_graph.draw()

        # Update Master Mask
        self.ax_mask.clear()
        self.ax_mask.imshow(master_mask, cmap='gray')
        self.ax_mask.axvline(x=centerline, color='red', linestyle='--', label='Centerline')
        # Highlight right side
        self.ax_mask.axvspan(centerline, master_mask.shape[1], color='red', alpha=0.3, label='Analyzed Region')
        self.ax_mask.set_title("Master Mask")
        self.ax_mask.legend(loc="upper right")
        self.canvas_mask.draw()

        self.run_btn.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = HalfToolAnalysisGUI()
    gui.show()
    sys.exit(app.exec())
