"""
Custom styled widgets for professional UI appearance.
"""
from PyQt6.QtWidgets import QCheckBox, QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush


class ToggleSwitch(QWidget):
    """iOS-style toggle switch for on/off states."""
    
    toggled = pyqtSignal(bool)  # Signal emitted when toggle state changes
    
    def __init__(self, parent=None, label_left="", label_right=""):
        super().__init__(parent)
        self.setFixedSize(60, 30)
        self._checked = False
        self._circle_pos = 3
        
        self.label_left = label_left
        self.label_right = label_right
        
        # Animation for smooth toggle
        self.animation = QPropertyAnimation(self, b"circle_position")
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.setDuration(200)
    
    @pyqtProperty(int)
    def circle_position(self):
        return self._circle_pos
    
    @circle_position.setter
    def circle_position(self, pos):
        self._circle_pos = pos
        self.update()
    
    def mousePressEvent(self, event):
        self.toggle()
    
    def toggle(self):
        self._checked = not self._checked
        
        start_pos = 3 if not self._checked else 33
        end_pos = 33 if self._checked else 3
        
        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.start()
        
        # Emit signal after animation starts
        self.toggled.emit(self._checked)
    
    def isChecked(self):
        return self._checked
    
    def setChecked(self, checked):
        if self._checked != checked:
            self.toggle()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background track
        if self._checked:
            bg_color = QColor(76, 175, 80)  # Green
        else:
            bg_color = QColor(189, 189, 189)  # Gray
        
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 5, 60, 20, 10, 10)
        
        # Draw circle thumb
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(self._circle_pos, 3, 24, 24)


class CircleCheckBox(QCheckBox):
    """Circular checkbox that fills when checked."""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                spacing: 8px;
                color: #FFFFFF;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #AAAAAA;
                border-radius: 10px;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4CAF50;
                border-radius: 10px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5, stop:0 #4CAF50, stop:1 #45a049);
            }
            QCheckBox::indicator:unchecked:hover {
                border: 2px solid #FFFFFF;
            }
            QCheckBox::indicator:checked:hover {
                border: 2px solid #66BB6A;
            }
        """)


class ToggleSwitchWithLabels(QWidget):
    """Toggle switch with labels on both sides."""
    
    toggled = pyqtSignal(bool)  # Forward the toggled signal
    
    def __init__(self, label_left="Raw", label_right="Processed", parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        self.left_label = QLabel(label_left)
        self.left_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        
        self.switch = ToggleSwitch()
        
        self.right_label = QLabel(label_right)
        self.right_label.setStyleSheet("color: #AAAAAA;")
        
        layout.addWidget(self.left_label)
        layout.addWidget(self.switch)
        layout.addWidget(self.right_label)
        layout.addStretch()
        
        # Connect signals
        self.switch.animation.finished.connect(self._update_labels)
        self.switch.toggled.connect(self.toggled.emit)  # Forward the signal
    
    def _update_labels(self):
        """Update label colors based on switch state."""
        if self.switch.isChecked():
            self.left_label.setStyleSheet("color: #AAAAAA;")
            self.right_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        else:
            self.left_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
            self.right_label.setStyleSheet("color: #AAAAAA;")
    
    def isChecked(self):
        return self.switch.isChecked()
    
    def setChecked(self, checked):
        self.switch.setChecked(checked)
        self._update_labels()
    
    def blockSignals(self, block):
        """Block signals on the internal switch."""
        return self.switch.blockSignals(block)
