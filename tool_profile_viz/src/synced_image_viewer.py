import imageio
import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QPushButton, QCheckBox
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt6.QtCore import Qt, pyqtSignal

# ... (_ImageView class is unchanged) ...
class _ImageView(QGraphicsView):
    viewChanged = pyqtSignal()
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)
    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0: self.scale(zoom_factor, zoom_factor)
        else: self.scale(1 / zoom_factor, 1 / zoom_factor)
        self.viewChanged.emit()
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag: self.viewChanged.emit()

class SyncedImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        
        self.raw_scene = QGraphicsScene(self)
        self.mask_scene = QGraphicsScene(self)
        self.raw_pixmap_item = self.raw_scene.addPixmap(QPixmap())
        self.mask_pixmap_item = self.mask_scene.addPixmap(QPixmap())

        self.raw_view = _ImageView(self.raw_scene)
        self.mask_view = _ImageView(self.mask_scene)
        
        # ROI line properties
        self.show_roi_line = False
        self.roi_height = 200  # Default, will be updated from metadata
        self.current_mask_path = None
        
        self.sync_checkbox = QCheckBox("Sync Zoom/Pan")
        self.sync_checkbox.setChecked(True)
        self.sync_checkbox.stateChanged.connect(self._toggle_sync)
        self.sync_enabled = True

        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_views)

        # --- NEW: Add a label for keyboard hints ---
        keyboard_hint_label = QLabel("Use ← → keys to cycle frames")
        keyboard_hint_label.setStyleSheet("font-style: italic; color: #AAAAAA;")

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.sync_checkbox)
        controls_layout.addWidget(keyboard_hint_label) # Add the hint
        controls_layout.addStretch()
        controls_layout.addWidget(reset_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("Raw Image"))
        main_layout.addWidget(self.raw_view)
        main_layout.addWidget(QLabel("Processed Mask"))
        main_layout.addWidget(self.mask_view)
        main_layout.addLayout(controls_layout)

        self._is_syncing = False
        self.raw_view.viewChanged.connect(lambda: self._sync_views(self.raw_view, self.mask_view))
        self.mask_view.viewChanged.connect(lambda: self._sync_views(self.mask_view, self.raw_view))
    
    def set_roi_line(self, show, roi_height=200):
        """Enable/disable ROI line and set the ROI height."""
        self.show_roi_line = show
        self.roi_height = roi_height
        if self.current_mask_path:
            self._reload_mask_with_roi()
    
    def _reload_mask_with_roi(self):
        """Reload the mask image with or without ROI line."""
        if not self.current_mask_path:
            return
            
        mask_tiff = imageio.imread(self.current_mask_path)
        
        if self.show_roi_line:
            # Calculate ROI boundary
            mask_np = np.array(mask_tiff)
            white_pixel_coords = np.where(mask_np == 255)
            
            if white_pixel_coords[0].size > 0:
                last_row = white_pixel_coords[0].max()
                roi_start_row = max(0, last_row - self.roi_height)
                
                # Draw red line at ROI boundary
                mask_with_line = mask_tiff.copy()
                if roi_start_row < mask_tiff.shape[0]:
                    mask_with_line[roi_start_row, :] = 255  # Draw horizontal line
                
                mask_qimage = QImage(mask_with_line.data, mask_with_line.shape[1], mask_with_line.shape[0], 
                                    mask_with_line.strides[0], QImage.Format.Format_Grayscale8)
            else:
                mask_qimage = QImage(mask_tiff.data, mask_tiff.shape[1], mask_tiff.shape[0], 
                                    mask_tiff.strides[0], QImage.Format.Format_Grayscale8)
        else:
            mask_qimage = QImage(mask_tiff.data, mask_tiff.shape[1], mask_tiff.shape[0], 
                                mask_tiff.strides[0], QImage.Format.Format_Grayscale8)
        
        # Create a red-tinted version for the ROI line
        if self.show_roi_line:
            pixmap = QPixmap.fromImage(mask_qimage)
            painter = QPainter(pixmap)
            pen = QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            
            # Recalculate ROI line position
            mask_np = np.array(mask_tiff)
            white_pixel_coords = np.where(mask_np == 255)
            if white_pixel_coords[0].size > 0:
                last_row = white_pixel_coords[0].max()
                roi_start_row = max(0, last_row - self.roi_height)
                painter.drawLine(0, roi_start_row, pixmap.width(), roi_start_row)
            painter.end()
            self.mask_pixmap_item.setPixmap(pixmap)
        else:
            self.mask_pixmap_item.setPixmap(QPixmap.fromImage(mask_qimage))

    # ... (rest of the file is unchanged) ...
    def _toggle_sync(self, state):
        self.sync_enabled = (state == Qt.CheckState.Checked.value)
    def reset_views(self):
        self.raw_view.fitInView(self.raw_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.mask_view.fitInView(self.mask_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    def _sync_views(self, source, target):
        if self._is_syncing or not self.sync_enabled: return
        self._is_syncing = True
        visible_rect = source.mapToScene(source.viewport().rect()).boundingRect()
        target.fitInView(visible_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._is_syncing = False
    def load_images(self, raw_path, mask_path):
        self.current_mask_path = mask_path  # Store for reloading
        
        raw_tiff = imageio.imread(raw_path)
        raw_qimage = QImage(raw_tiff.data, raw_tiff.shape[1], raw_tiff.shape[0], raw_tiff.strides[0], QImage.Format.Format_RGB888)
        self.raw_pixmap_item.setPixmap(QPixmap.fromImage(raw_qimage))
        
        self._reload_mask_with_roi()
        self.reset_views()