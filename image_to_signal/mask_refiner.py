import sys
import os
import glob
import re
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QListWidget, QSlider, QFileDialog,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QCheckBox,
                             QGraphicsEllipseItem)
from PyQt6.QtCore import Qt, QPoint, QRectF, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QPainterPath

def extract_frame_num(filepath):
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    match = re.match(r"^(\d+\.?\d*)", name)
    if match: return float(match.group(1))
    parts = name.split("_")
    for part in reversed(parts):
        try: return float(part)
        except ValueError: continue
    return 0.0

def load_as_rgb32_qimage(path):
    from PIL import Image
    try:
        img = Image.open(path).convert('RGB')
        data = img.tobytes("raw", "RGB")
        qim = QImage(data, img.size[0], img.size[1], QImage.Format.Format_RGB888)
        return qim.convertToFormat(QImage.Format.Format_RGB32)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return QImage()

class DrawableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.current_qimage = None
        self.current_pixmap_item = None
        self.overlay_qimage = None
        self.overlay_pixmap_item = None
        
        self.drawing = False
        self.last_point = None
        self.brush_size = 10
        self.draw_color = Qt.GlobalColor.white # white for fill, black for erase
        self.opacity = 0.5
        
        self.parent_ref = None # Will point to MainWindow to notify changes
        
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setPen(QPen(Qt.GlobalColor.red, 1))
        self.cursor_item.setZValue(10)
        self.cursor_item.hide()
        self.scene.addItem(self.cursor_item)
        
    def load_images(self, current_path, overlay_path=None):
        if self.current_pixmap_item:
            self.scene.removeItem(self.current_pixmap_item)
        if self.overlay_pixmap_item:
            self.scene.removeItem(self.overlay_pixmap_item)
            
        self.current_qimage = load_as_rgb32_qimage(current_path)
        self.current_pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.current_qimage))
        self.scene.addItem(self.current_pixmap_item)
        
        if overlay_path and os.path.exists(overlay_path):
            overlay_orig = load_as_rgb32_qimage(overlay_path)
            self.overlay_qimage = QImage(overlay_orig.size(), QImage.Format.Format_ARGB32)
            self.overlay_qimage.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(self.overlay_qimage)
            for y in range(overlay_orig.height()):
                for x in range(overlay_orig.width()):
                    if QColor(overlay_orig.pixel(x, y)).red() > 127:
                        self.overlay_qimage.setPixelColor(x, y, QColor(255, 0, 0, int(255 * self.opacity)))
            painter.end()
            
            self.overlay_pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.overlay_qimage))
            self.scene.addItem(self.overlay_pixmap_item)
            self.overlay_pixmap_item.setZValue(1)
        else:
            self.overlay_qimage = None
            self.overlay_pixmap_item = None
            
        self.setSceneRect(QRectF(self.current_qimage.rect()))
        
    def update_overlay_opacity(self, opacity):
        self.opacity = opacity
        if self.overlay_pixmap_item:
            self.overlay_pixmap_item.setOpacity(opacity)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
            self.drawing = True
            self.draw_color = Qt.GlobalColor.white if event.button() == Qt.MouseButton.LeftButton else Qt.GlobalColor.black
            pos = self.mapToScene(event.pos())
            self.last_point = pos
            self.draw_on_image(pos)
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        radius = self.brush_size / 2.0
        self.cursor_item.setRect(QRectF(pos.x() - radius, pos.y() - radius, self.brush_size, self.brush_size))
        self.cursor_item.show()
        
        if self.drawing and self.last_point:
            self.draw_on_image(pos, self.last_point)
            self.last_point = pos
        super().mouseMoveEvent(event)
        
    def leaveEvent(self, event):
        self.cursor_item.hide()
        super().leaveEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
            self.drawing = False
            self.last_point = None
            if self.parent_ref:
                self.parent_ref.mark_unsaved()
        super().mouseReleaseEvent(event)
        
    def draw_on_image(self, current_pos, last_pos=None):
        if not self.current_qimage: return
        
        painter = QPainter(self.current_qimage)
        pen = QPen(self.draw_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        if last_pos:
            painter.drawLine(last_pos, current_pos)
        else:
            painter.drawPoint(current_pos)
            
        painter.end()
        
        # Update pixmap
        self.current_pixmap_item.setPixmap(QPixmap.fromImage(self.current_qimage))
        
    def wheelEvent(self, event):
        # Zooming
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        
    def reset_zoom(self):
        self.resetTransform()
        
    def save_current(self, path):
        if self.current_qimage:
            import numpy as np
            from PIL import Image
            qim = self.current_qimage.convertToFormat(QImage.Format.Format_Grayscale8)
            width = qim.width()
            height = qim.height()
            bpl = qim.bytesPerLine()
            data = qim.constBits().asstring(height * bpl)
            arr = np.frombuffer(data, dtype=np.uint8).reshape((height, bpl))
            arr = arr[:, :width]
            out_img = Image.fromarray(arr, mode='L')
            out_img.save(path)

class MaskRefiner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Refiner - Custom Brush & Overlay")
        self.resize(1200, 800)
        
        self.folder_path = ""
        self.files = []
        self.current_file = None
        self.reference_file = None
        self.unsaved_changes = False
        
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        
        btn_browse = QPushButton("📁 Browse Masks Folder")
        btn_browse.clicked.connect(self.browse_folder)
        left_layout.addWidget(btn_browse)
        
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.on_file_selected)
        left_layout.addWidget(self.list_widget)
        
        self.lbl_current = QLabel("Current: None")
        left_layout.addWidget(self.lbl_current)
        
        btn_set_ref = QPushButton("📌 Set Current as Reference")
        btn_set_ref.clicked.connect(self.set_reference)
        left_layout.addWidget(btn_set_ref)
        
        btn_clear_ref = QPushButton("❌ Clear Reference")
        btn_clear_ref.clicked.connect(self.clear_reference)
        left_layout.addWidget(btn_clear_ref)
        
        self.lbl_ref = QLabel("Reference: None")
        left_layout.addWidget(self.lbl_ref)
        
        # Opacity
        left_layout.addWidget(QLabel("Overlay Opacity:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        self.slider_opacity.valueChanged.connect(self.on_opacity_changed)
        left_layout.addWidget(self.slider_opacity)
        
        # Brush Size
        left_layout.addWidget(QLabel("Brush Size:"))
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(1, 100)
        self.slider_brush.setValue(10)
        self.slider_brush.valueChanged.connect(self.on_brush_changed)
        left_layout.addWidget(self.slider_brush)
        
        btn_reset_zoom = QPushButton("🔍 Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        left_layout.addWidget(btn_reset_zoom)
        
        left_layout.addWidget(QLabel("Left Click: Draw White (Fill)\nRight Click: Draw Black (Erase)\nScroll: Zoom In/Out"))
        
        self.btn_save = QPushButton("💾 Save Mask")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_mask)
        left_layout.addWidget(self.btn_save)
        
        layout.addWidget(left_panel)
        
        # Right panel (Graphics View)
        self.view = DrawableGraphicsView()
        self.view.parent_ref = self
        layout.addWidget(self.view, stretch=1)
        
    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Masks Folder")
        if d:
            self.folder_path = d
            self.load_folder()
            
    def load_folder(self):
        self.list_widget.clear()
        self.files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif'):
            self.files.extend(glob.glob(os.path.join(self.folder_path, ext)))
        
        self.files.sort(key=extract_frame_num)
        
        for f in self.files:
            self.list_widget.addItem(os.path.basename(f))
            
    def on_file_selected(self, current, previous):
        if not current: return
        
        if self.unsaved_changes:
            reply = QMessageBox.question(self, 'Unsaved Changes', 
                                         'You have unsaved changes. Save before switching?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self.save_mask()
            elif reply == QMessageBox.StandardButton.Cancel:
                self.list_widget.blockSignals(True)
                if previous: self.list_widget.setCurrentItem(previous)
                self.list_widget.blockSignals(False)
                return
                
        self.current_file = os.path.join(self.folder_path, current.text())
        self.lbl_current.setText(f"Current: {current.text()}")
        self.unsaved_changes = False
        self.btn_save.setText("💾 Save Mask")
        
        self.view.load_images(self.current_file, self.reference_file)
        self.view.update_overlay_opacity(self.slider_opacity.value() / 100.0)
        self.view.brush_size = self.slider_brush.value()
        
    def set_reference(self):
        if self.current_file:
            self.reference_file = self.current_file
            self.lbl_ref.setText(f"Reference: {os.path.basename(self.reference_file)}")
            # Reload to show overlay
            self.view.load_images(self.current_file, self.reference_file)
            self.view.update_overlay_opacity(self.slider_opacity.value() / 100.0)

    def clear_reference(self):
        self.reference_file = None
        self.lbl_ref.setText("Reference: None")
        if self.current_file:
            self.view.load_images(self.current_file, None)
            self.view.update_overlay_opacity(self.slider_opacity.value() / 100.0)
            
    def on_opacity_changed(self, val):
        self.view.update_overlay_opacity(val / 100.0)
        
    def on_brush_changed(self, val):
        self.view.brush_size = val
        pos = self.view.mapToScene(self.view.mapFromGlobal(self.view.cursor().pos()))
        radius = val / 2.0
        self.view.cursor_item.setRect(QRectF(pos.x() - radius, pos.y() - radius, val, val))
        
    def reset_zoom(self):
        self.view.reset_zoom()
        
    def mark_unsaved(self):
        self.unsaved_changes = True
        self.btn_save.setText("💾 Save Mask *")
        
    def save_mask(self):
        if self.current_file:
            self.view.save_current(self.current_file)
            self.unsaved_changes = False
            self.btn_save.setText("💾 Save Mask")
            QMessageBox.information(self, "Saved", "Mask saved successfully.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MaskRefiner()
    window.show()
    sys.exit(app.exec())
