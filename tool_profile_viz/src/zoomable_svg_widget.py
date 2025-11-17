from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen

# We create an inner class to handle events cleanly without cluttering the main widget.
class _SvgView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # Degree indicator properties
        self.show_degree_indicator = False
        self.current_degree = None

    def set_degree_indicator(self, show, degree=None):
        """Enable/disable and update the degree indicator line."""
        self.show_degree_indicator = show
        self.current_degree = degree
        self.viewport().update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Override to draw degree indicator line on top of SVG."""
        super().paintEvent(event)
        
        if self.show_degree_indicator and self.current_degree is not None:
            # Map degree (0-360) to scene x-coordinate
            scene_rect = self.sceneRect()
            x_scene = scene_rect.left() + (self.current_degree / 360.0) * scene_rect.width()
            
            # Convert scene coordinates to viewport coordinates
            top_point = self.mapFromScene(x_scene, scene_rect.top())
            bottom_point = self.mapFromScene(x_scene, scene_rect.bottom())
            
            # Draw the red vertical line
            painter = QPainter(self.viewport())
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(top_point, bottom_point)
            painter.end()

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)

    def mousePressEvent(self, event):
        """Enable panning when the left mouse button is pressed."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Disable panning when the left mouse button is released."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

class ZoomableSvgWidget(QWidget):
    def __init__(self, svg_path):
        super().__init__()
        
        self.svg_path = svg_path  # Store for potential reloading
        
        # --- Create Graphics Components ---
        self.scene = QGraphicsScene(self)
        self.svg_item = QGraphicsSvgItem(svg_path)
        self.scene.addItem(self.svg_item)
        
        # Use our custom view class
        self.view = _SvgView(self.scene)
        
        # --- Create Overlay Controls ---
        self.home_button = QPushButton("Reset View")
        self.home_button.setFixedSize(80, 30)
        self.home_button.clicked.connect(self.reset_view)

        self.controls_label = QLabel("Scroll: Zoom | Left-Click+Drag: Pan")
        self.controls_label.setStyleSheet("""
            background-color: rgba(45, 62, 80, 0.8);
            color: white;
            padding: 5px;
            border-radius: 3px;
        """)
        
        # --- Layouts ---
        # Main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.view)

        # A container for the controls, parented to the view so it floats on top
        self.controls_container = QWidget(self.view)
        controls_layout = QHBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.addWidget(self.home_button)
        controls_layout.addWidget(self.controls_label)
        controls_layout.addStretch()
    
    def load_svg(self, svg_path):
        """Load a new SVG file."""
        self.svg_path = svg_path
        self.scene.removeItem(self.svg_item)
        self.svg_item = QGraphicsSvgItem(svg_path)
        self.scene.addItem(self.svg_item)
        self.reset_view()
    
    def set_degree_indicator(self, show, degree=None):
        """Pass degree indicator settings to the view."""
        self.view.set_degree_indicator(show, degree)

    def reset_view(self):
        """Resets the view to its initial zoom and pan."""
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        """Ensure the SVG fits and the controls are positioned correctly."""
        super().resizeEvent(event)
        self.reset_view()
        # Position the controls in the bottom-right corner of the view
        container_size = self.controls_container.sizeHint()
        view_rect = self.view.viewport().rect()
        self.controls_container.move(
            view_rect.right() - container_size.width() - 10,
            view_rect.bottom() - container_size.height() - 10
        )