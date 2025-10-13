from PyQt6.QtWidgets import QMainWindow, QApplication

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tool Profile Viz")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height