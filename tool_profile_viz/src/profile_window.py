import os
import re
import imageio
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from .zoomable_svg_widget import ZoomableSvgWidget
# ... (imports are the same)
from .zoomable_svg_widget import ZoomableSvgWidget

class ProfileWindow(QMainWindow):
    def __init__(self, tool_id, svg_path, overview_image_paths):
        super().__init__()
        # ... (init code is the same)
        self.tool_id = tool_id
        self.setWindowTitle(f"Profile View: {self.tool_id}")
        self.setGeometry(150, 150, 1400, 800) 

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_panel_layout = QVBoxLayout()
        
        # --- TWEAK: Simplified caption ---
        svg_caption = QLabel("1D Signal Profile")
        svg_caption.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.svg_widget = ZoomableSvgWidget(svg_path)
        
        # ... (the rest of the file is exactly the same as before)
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(0,0,0,0) # Remove internal margins
        overview_caption = QLabel("Quarter-Turn Overview")
        overview_caption.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        overview_images_layout = QHBoxLayout()
        
        for path in overview_image_paths:
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

        left_panel_layout.addWidget(svg_caption)
        left_panel_layout.addWidget(self.svg_widget)
        left_panel_layout.addWidget(overview_widget)

        right_panel_placeholder = QLabel("Placeholder for Interactive Frame Explorer")
        right_panel_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel_placeholder.setStyleSheet("border: 1px solid white; font-size: 16px;")

        main_layout.addLayout(left_panel_layout, 3) 
        main_layout.addWidget(right_panel_placeholder, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)