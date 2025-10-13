import sys
from PyQt6.QtWidgets import QApplication
from src.main_window import MainWindow

# --- Application Style Sheet ---
APP_STYLE = """
    /* (Previous styles are unchanged) */
    QMainWindow, QWidget {
        background-color: #1A2A3A; color: #FFFFFF;
    }
    QTableView, QTableWidget {
        background-color: #2D3E50; gridline-color: #1A2A3A;
    }
    QHeaderView::section {
        background-color: #1A2A3A; color: #FFFFFF; padding: 4px; border: 1px solid #2D3E50;
    }
    QPushButton#TopControlButton {
        background-color: #005A9C; color: white; padding: 5px 10px;
        border-radius: 3px; min-width: 100px;
    }
    QPushButton#TopControlButton:hover {
        background-color: #007ACC;
    }
    /* --- NEW: Style for the Save button when there are unsaved changes --- */
    QPushButton#TopControlButtonDirty {
        background-color: #D2691E; /* Amber/Chocolate Color */
        color: white; padding: 5px 10px; border-radius: 3px; min-width: 100px;
        font-weight: bold; border: 1px solid #FFFFFF;
    }
    QPushButton#ActionButtonSuccess {
        background-color: #556B2F; color: white; font-weight: bold;
    }
    QPushButton#ActionButtonFailure {
        background-color: #8B0000; color: #CCCCCC;
    }
    QPushButton#DeleteButton {
        background-color: #404040; color: #FFFFFF;
    }
    QPushButton#DeleteButton:hover {
        background-color: #8B0000;
    }
    QLabel { /* Ensures the reminder and zoom labels are white */
        color: #FFFFFF;
    }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())