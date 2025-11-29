import os
import re
import json
import imageio
from PyQt6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLineEdit)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QEvent
from .matplotlib_plot_widget import MatplotlibPlotWidget
from .synced_image_viewer import SyncedImageViewer
from .custom_widgets import ToggleSwitchWithLabels, CircleCheckBox

class ProfileWindow(QMainWindow):
    def __init__(self, tool_id, svg_path, overview_paths, blurred_folder, mask_folder):
        super().__init__()
        
        self.tool_id = tool_id
        self.blurred_folder = blurred_folder
        self.mask_folder = mask_folder
        
        # Determine CSV paths
        profiles_dir = os.path.dirname(svg_path)
        self.raw_csv_path = os.path.join(profiles_dir, f'{tool_id}_raw_data.csv')
        self.processed_csv_path = os.path.join(profiles_dir, f'{tool_id}_processed_data.csv')
        
        # Load metadata
        self.metadata = self._load_metadata(profiles_dir)
        self.roi_height = self.metadata.get('roi_parameters', {}).get('roi_height', 200)
        
        self.setWindowTitle(f"Profile View: {self.tool_id}")
        self.setGeometry(150, 150, 1600, 900) 

        self._create_image_maps()
        self.current_degree_index = 0  # Start at first image

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_panel = self._create_left_panel(overview_paths)
        right_panel = self._create_right_panel()

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.showMaximized()
        
        # Auto-load first image
        if self.sorted_degrees:
            self._show_frames_for_degree(self.sorted_degrees[0])
    
    def _load_metadata(self, profiles_dir):
        """Load metadata JSON for this tool."""
        metadata_dir = os.path.join(profiles_dir, 'analysis_metadata')
        metadata_path = os.path.join(metadata_dir, f'{self.tool_id}_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {}  # Return empty dict if not found

    # --- FIX: Use an event filter to capture key presses from the input box ---
    def eventFilter(self, source, event):
        """Intercepts key presses from the degree input field."""
        if source is self.degree_input and event.type() == QEvent.Type.KeyPress:
            if not self.sorted_degrees:
                return super().eventFilter(source, event)

            key = event.key()
            num_degrees = len(self.sorted_degrees)

            if key == Qt.Key.Key_Right:
                self.current_degree_index = (self.current_degree_index + 1) % num_degrees
                self._show_frames_for_degree(self.sorted_degrees[self.current_degree_index])
                return True # Event handled, don't pass to input box
            elif key == Qt.Key.Key_Left:
                self.current_degree_index = (self.current_degree_index - 1 + num_degrees) % num_degrees
                self._show_frames_for_degree(self.sorted_degrees[self.current_degree_index])
                return True # Event handled

        # For all other events, let the default handler run
        return super().eventFilter(source, event)

    def _create_image_maps(self):
        print(f"[DEBUG] Creating image maps for tool {self.tool_id}")
        print(f"[DEBUG] Blurred folder: {self.blurred_folder}")
        print(f"[DEBUG] Mask folder: {self.mask_folder}")
        print(f"[DEBUG] Blurred folder exists: {os.path.exists(self.blurred_folder)}")
        print(f"[DEBUG] Mask folder exists: {os.path.exists(self.mask_folder)}")
        
        self.raw_image_map = {}
        if os.path.exists(self.blurred_folder):
            files = os.listdir(self.blurred_folder)
            print(f"[DEBUG] Found {len(files)} files in blurred folder")
            for f in files:
                match = re.search(r"(\d{4}\.\d{2})", f)
                if match:
                    self.raw_image_map[float(match.group(1))] = os.path.join(self.blurred_folder, f)
            print(f"[DEBUG] Mapped {len(self.raw_image_map)} blurred images")
        else:
            print(f"[DEBUG] Blurred folder does not exist!")
        
        self.mask_image_map = {}
        if os.path.exists(self.mask_folder):
            files = os.listdir(self.mask_folder)
            print(f"[DEBUG] Found {len(files)} files in mask folder")
            for f in files:
                match = re.search(r"(\d{4}\.\d{2})", f)
                if match:
                    self.mask_image_map[float(match.group(1))] = os.path.join(self.mask_folder, f)
            print(f"[DEBUG] Mapped {len(self.mask_image_map)} mask images")
        else:
            print(f"[DEBUG] Mask folder does not exist!")
            
        self.sorted_degrees = sorted(self.raw_image_map.keys())
        print(f"[DEBUG] Total sorted degrees: {len(self.sorted_degrees)}")

    def _create_left_panel(self, overview_paths):
        left_panel_widget = QWidget()
        layout = QVBoxLayout(left_panel_widget)
        
        # Controls header
        controls_layout = QHBoxLayout()
        
        # Graph type toggle
        self.graph_toggle = ToggleSwitchWithLabels("Raw", "Processed")
        self.graph_toggle.toggled.connect(self._on_graph_type_changed)
        controls_layout.addWidget(QLabel("Graph Type:"))
        controls_layout.addWidget(self.graph_toggle)
        controls_layout.addSpacing(20)
        
        # Degree indicator checkbox
        self.degree_indicator_checkbox = CircleCheckBox("Show Current Degree")
        self.degree_indicator_checkbox.setChecked(True)
        self.degree_indicator_checkbox.stateChanged.connect(self._on_degree_indicator_toggled)
        controls_layout.addWidget(self.degree_indicator_checkbox)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Matplotlib plot widget - check for processed first
        if os.path.exists(self.processed_csv_path):
            self.plot_widget = MatplotlibPlotWidget(self.processed_csv_path, is_processed=True)
            self.graph_toggle.setChecked(True)  # Set to processed
        else:
            self.plot_widget = MatplotlibPlotWidget(self.raw_csv_path, is_processed=False)
        
        self.plot_widget.setMaximumHeight(700)  # Limit plot height
        layout.addWidget(self.plot_widget, stretch=5)
        
        # Overview section
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(5, 10, 5, 5)
        overview_caption = QLabel("Quarter-Turn Overview")
        overview_caption.setStyleSheet("font-weight: bold; font-size: 14px;")
        overview_layout.addWidget(overview_caption)
        
        overview_images_layout = QHBoxLayout()
        overview_images_layout.setSpacing(10)
        
        for path in overview_paths:
            img_layout = QVBoxLayout()
            img_layout.setSpacing(2)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            try:
                tiff_image = imageio.imread(path)
                q_image = QImage(tiff_image.data, tiff_image.shape[1], tiff_image.shape[0], 
                                tiff_image.strides[0], QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, 
                                             Qt.TransformationMode.SmoothTransformation)
                img_label.setPixmap(scaled_pixmap)
            except Exception as e:
                img_label.setText(f"Error loading\n{os.path.basename(path)}")
                img_label.setStyleSheet("color: red;")
            
            degree_match = re.search(r"(\d{4}\.\d{2})", os.path.basename(path))
            degree_text = f"{float(degree_match.group(1))}°" if degree_match else "N/A"
            degree_label = QLabel(degree_text)
            degree_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            degree_label.setStyleSheet("font-size: 12px;")
            
            img_layout.addWidget(img_label)
            img_layout.addWidget(degree_label)
            overview_images_layout.addLayout(img_layout)
        
        overview_layout.addLayout(overview_images_layout)
        overview_widget.setMinimumHeight(220)
        overview_widget.setMaximumHeight(280)
        layout.addWidget(overview_widget, stretch=1)
        return left_panel_widget
    
    def _on_graph_type_changed(self, is_processed):
        """Switch between raw and processed graph."""
        csv_path = self.processed_csv_path if is_processed else self.raw_csv_path
        if os.path.exists(csv_path):
            self.plot_widget.reload_csv(csv_path)
            self._update_degree_indicator()
        else:
            # Reset toggle if file doesn't exist
            self.graph_toggle.blockSignals(True)
            self.graph_toggle.setChecked(False)
            self.graph_toggle.blockSignals(False)
    
    def _on_degree_indicator_toggled(self):
        """Toggle degree indicator on/off."""
        self._update_degree_indicator()
    
    def _update_degree_indicator(self):
        """Update the degree indicator line on the graph."""
        show = self.degree_indicator_checkbox.isChecked()
        degree = self.sorted_degrees[self.current_degree_index] if self.current_degree_index >= 0 else None
        
        self.plot_widget.set_degree_indicator(show, degree)
    
    def _create_right_panel(self):
        right_panel_widget = QWidget()
        layout = QVBoxLayout(right_panel_widget)
        
        # Input controls and image counter
        input_layout = QHBoxLayout()
        input_label = QLabel("Enter Degree:")
        self.degree_input = QLineEdit()
        self.degree_input.setPlaceholderText("e.g., 125.5")
        self.degree_input.returnPressed.connect(self._on_degree_input)
        
        # --- FIX: Install the event filter on the input box ---
        self.degree_input.installEventFilter(self)

        input_layout.addWidget(input_label)
        input_layout.addWidget(self.degree_input)
        
        # Image counter
        self.image_counter_label = QLabel("Image: 1/360")
        self.image_counter_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        input_layout.addWidget(self.image_counter_label)
        
        # ROI line toggle
        self.roi_line_checkbox = CircleCheckBox("Show ROI Line on Mask")
        self.roi_line_checkbox.setChecked(False)
        self.roi_line_checkbox.stateChanged.connect(self._on_roi_line_changed)
        
        self.image_caption = QLabel("Showing frames for: N/A")
        self.image_caption.setStyleSheet("font-style: italic;")
        self.image_viewer = SyncedImageViewer()
        
        layout.addLayout(input_layout)
        layout.addWidget(self.roi_line_checkbox)
        layout.addWidget(self.image_caption)
        layout.addWidget(self.image_viewer)
        return right_panel_widget
    
    def _on_roi_line_changed(self):
        """Toggle ROI line on mask image."""
        show = self.roi_line_checkbox.isChecked()
        self.image_viewer.set_roi_line(show, self.roi_height)

    def _on_degree_input(self):
        try:
            target_deg = float(self.degree_input.text())
        except ValueError:
            self.image_caption.setText("Invalid input. Please enter a number.")
            return

        if not self.sorted_degrees:
            self.image_caption.setText("No images found.")
            return
            
        closest_deg = min(self.sorted_degrees, key=lambda d: abs(d - target_deg))
        self._show_frames_for_degree(closest_deg)

    def _show_frames_for_degree(self, degree):
        print(f"[DEBUG] Showing frames for degree: {degree}")
        raw_path = self.raw_image_map.get(degree)
        mask_path = self.mask_image_map.get(degree)
        print(f"[DEBUG] Raw path: {raw_path}")
        print(f"[DEBUG] Mask path: {mask_path}")
        print(f"[DEBUG] Raw path exists: {os.path.exists(raw_path) if raw_path else 'N/A'}")
        print(f"[DEBUG] Mask path exists: {os.path.exists(mask_path) if mask_path else 'N/A'}")

        if raw_path and mask_path:
            self.image_caption.setText(f"Showing frames for: {degree}°")
            self.degree_input.setText(str(degree))
            print(f"[DEBUG] Calling image_viewer.load_images({raw_path}, {mask_path})")
            self.image_viewer.load_images(raw_path, mask_path)
            self.current_degree_index = self.sorted_degrees.index(degree)
            
            # Update image counter
            total_images = len(self.sorted_degrees)
            current_index = self.current_degree_index + 1  # 1-indexed for display
            self.image_counter_label.setText(f"Image: {current_index}/{total_images}")
            
            # Update degree indicator on graph
            self._update_degree_indicator()
        else:
            print(f"[DEBUG] Missing frame(s) for {degree}°")
            self.image_caption.setText(f"Missing a frame for {degree}°")