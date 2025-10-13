import imageio
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QPushButton, QCheckBox
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal

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
        
        self.sync_checkbox = QCheckBox("Sync Zoom/Pan")
        self.sync_checkbox.setChecked(True)
        self.sync_checkbox.stateChanged.connect(self._toggle_sync)
        self.sync_enabled = True

        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_views)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.sync_checkbox)
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

    def _toggle_sync(self, state):
        self.sync_enabled = (state == Qt.CheckState.Checked.value)

    def reset_views(self):
        self.raw_view.fitInView(self.raw_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.mask_view.fitInView(self.mask_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _sync_views(self, source, target):
        if self._is_syncing or not self.sync_enabled:
            return
        self._is_syncing = True
        target.setTransform(source.transform())
        self._is_syncing = False

    def load_images(self, raw_path, mask_path):
        raw_tiff = imageio.imread(raw_path)
        raw_qimage = QImage(raw_tiff.data, raw_tiff.shape[1], raw_tiff.shape[0], raw_tiff.strides[0], QImage.Format.Format_RGB888)
        self.raw_pixmap_item.setPixmap(QPixmap.fromImage(raw_qimage))
        
        mask_tiff = imageio.imread(mask_path)
        # --- THIS IS THE FIX ---
        # Use Grayscale8 for single-channel mask images, not RGB888.
        mask_qimage = QImage(mask_tiff.data, mask_tiff.shape[1], mask_tiff.shape[0], mask_tiff.strides[0], QImage.Format.Format_Grayscale8)
        self.mask_pixmap_item.setPixmap(QPixmap.fromImage(mask_qimage))
        
        self.reset_views()