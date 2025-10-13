from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtCore import Qt

class ZoomableSvgWidget(QGraphicsView):
    def __init__(self, svg_path):
        super().__init__()
        
        # Create a scene to hold the SVG item
        self.scene = QGraphicsScene(self)
        self.svg_item = QGraphicsSvgItem(svg_path)
        
        # Add the SVG to the scene
        self.scene.addItem(self.svg_item)
        self.setScene(self.scene)
        
        # Configure the view
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.scale(1 / zoom_factor, 1 / zoom_factor)

    def resizeEvent(self, event):
        """Ensure the SVG fits the view while maintaining aspect ratio."""
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)