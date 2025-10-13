import sys
from PyQt6.QtWidgets import QApplication
from src.main_window import MainWindow

# --- Application Style Sheet ---
# Based on your preferences: navy/black, white text, olive/leaf green accents.
APP_STYLE = """
    QMainWindow, QWidget {
        background-color: #1A2A3A; /* Dark Navy Blue */
        color: #FFFFFF; /* White Text */
    }
    QTableView, QTableWidget {
        background-color: #2D3E50; /* Slightly Lighter Navy */
        gridline-color: #1A2A3A;
    }
    QHeaderView::section {
        background-color: #1A2A3A;
        color: #FFFFFF;
        padding: 4px;
        border: 1px solid #2D3E50;
    }
    QPushButton#SaveButton {
        background-color: #005A9C; /* A complementary blue */
        color: white;
        padding: 5px;
        border-radius: 3px;
    }
    QPushButton#ActionButtonSuccess {
        background-color: #556B2F; /* Olive Green */
        color: white;
        font-weight: bold;
    }
    QPushButton#ActionButtonFailure {
        background-color: #8B0000; /* Dark Red */
        color: #CCCCCC; /* Light gray text */
    }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE) # Apply the theme to the whole app
    window = MainWindow()
    window.show()
    sys.exit(app.exec())