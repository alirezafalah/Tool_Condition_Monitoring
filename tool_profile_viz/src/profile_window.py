import os
import re
import imageio
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from .zoomable_svg_widget import ZoomableSvgWidget
from .synced_image_viewer import SyncedImageViewer

class ProfileWindow(QMainWindow):
    def __init__(self, tool_id, svg_path, overview_paths, blurred_folder, mask_folder):
        super().__init__()
        
        self.tool_id = tool_id
        self.blurred_folder = blurred_folder
        self.mask_folder = mask_folder
        self.setWindowTitle(f"Profile View: {self.tool_id}")
        self.setGeometry(150, 150, 1600, 900)
        self._create_image_maps()

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_panel = self._create_left_panel(svg_path, overview_paths)
        right_panel = self._create_right_panel()

        # --- TWEAK: Adjusted stretch factors for better balance ---
        main_layout.addWidget(left_panel, 3)  # Left side takes 3/5 (60%)
        main_layout.addWidget(right_panel, 2) # Right side takes 2/5 (40%)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.showMaximized()

    def _create_image_maps(self):
        """Creates dictionaries mapping degrees to full file paths."""
        self.raw_image_map = {}
        for f in os.listdir(self.blurred_folder):
            match = re.search(r"(\d{4}\.\d{2})", f)
            if match:
                self.raw_image_map[float(match.group(1))] = os.path.join(self.blurred_folder, f)
        
        self.mask_image_map = {}
        for f in os.listdir(self.mask_folder):
            match = re.search(r"(\d{4}\.\d{2})", f)
            if match:
                self.mask_image_map[float(match.group(1))] = os.path.join(self.mask_folder, f)

    def _create_left_panel(self, svg_path, overview_paths):
        """
        FIX: This function now contains the complete UI code for the left panel.
        """
        left_panel_widget = QWidget()
        layout = QVBoxLayout(left_panel_widget)

        # SVG Viewer
        svg_caption = QLabel("1D Signal Profile")
        svg_caption.setStyleSheet("font-weight: bold; font-size: 14px;")
        svg_viewer = ZoomableSvgWidget(svg_path)
        
        # Overview Images Widget
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(0,0,0,0)
        overview_caption = QLabel("Quarter-Turn Overview: Visual inspection at 90° increments.")
        overview_caption.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        overview_images_layout = QHBoxLayout()
        
        for path in overview_paths:
            img_layout = QVBoxLayout()
            img_label = QLabel()
            
            tiff_image = imageio.imread(path)
            q_image = QImage(tiff_image.data, tiff_image.shape[1], tiff_image.shape[0], tiff_image.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            img_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            
            degree_match = re.search(r"(\d{4}\.\d{2})", os.path.basename(path))
            degree_text = f"{float(degree_match.group(1))}°" if degree_match else "N/A"
            degree_label = QLabel(degree_text)
            degree_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            img_layout.addWidget(img_label)
            img_layout.addWidget(degree_label)
            overview_images_layout.addLayout(img_layout)
        
        overview_layout.addWidget(overview_caption)
        overview_layout.addLayout(overview_images_layout)
        overview_widget.setMaximumHeight(280)

        layout.addWidget(svg_caption)
        layout.addWidget(svg_viewer)
        layout.addWidget(overview_widget)
        
        return left_panel_widget
    
    def _create_right_panel(self):
        """Creates the interactive frame explorer panel."""
        right_panel_widget = QWidget()
        layout = QVBoxLayout(right_panel_widget)
        
        input_layout = QHBoxLayout()
        input_label = QLabel("Enter Degree:")
        self.degree_input = QLineEdit()
        self.degree_input.setPlaceholderText("e.g., 125.5")
        self.degree_input.returnPressed.connect(self._on_degree_selected)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.degree_input)
        
        self.image_caption = QLabel("Showing frames for: N/A")
        self.image_caption.setStyleSheet("font-style: italic;")

        self.image_viewer = SyncedImageViewer()

        layout.addLayout(input_layout)
        layout.addWidget(self.image_caption)
        layout.addWidget(self.image_viewer)
        return right_panel_widget

    def _on_degree_selected(self):
        """Finds and displays the images for the selected degree."""
        try:
            target_deg = float(self.degree_input.text())
        except ValueError:
            self.image_caption.setText("Invalid input. Please enter a number.")
            return

        if not self.raw_image_map:
            self.image_caption.setText("No images found.")
            return
            
        closest_deg = min(self.raw_image_map.keys(), key=lambda d: abs(d - target_deg))
        
        raw_path = self.raw_image_map.get(closest_deg)
        mask_path = self.mask_image_map.get(closest_deg)

        if raw_path and mask_path:
            self.image_caption.setText(f"Showing frames for: {closest_deg}°")
            self.image_viewer.load_images(raw_path, mask_path)
        else:
            self.image_caption.setText(f"Missing a frame for {closest_deg}°")