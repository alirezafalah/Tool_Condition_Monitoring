import os
from PyQt6.QtWidgets import (QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, 
                             QWidget, QPushButton, QMessageBox, QHeaderView)
from PyQt6.QtGui import QColor
from .metadata_manager import MetadataManager # Import the new manager

# The path to the data folder is needed to check for SVG files.
DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Use the new, more descriptive window title
        self.setWindowTitle("Tool Profile Visualization - Tool Metadata")
        self.setGeometry(100, 100, 1000, 800)

        self.metadata_manager = MetadataManager()
        self.tool_data = self.metadata_manager.get_all_tools()

        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(7) # Reduced columns as we abstract
        self.table.setHorizontalHeaderLabels(["Tool ID", "Type", "Diameter (mm)", "Edges", "Condition", "Notes", "Profile"])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.populate_table()
        layout.addWidget(self.table)

        save_button = QPushButton("Save Changes to Metadata")
        save_button.setObjectName("SaveButton") # For styling
        save_button.clicked.connect(self.save_metadata_from_table)
        layout.addWidget(save_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def populate_table(self):
        self.table.setRowCount(len(self.tool_data))
        for row_index, tool in enumerate(self.tool_data):
            self.table.setItem(row_index, 0, QTableWidgetItem(tool.get("tool_id", "")))
            self.table.setItem(row_index, 1, QTableWidgetItem(tool.get("type", "")))
            self.table.setItem(row_index, 2, QTableWidgetItem(str(tool.get("diameter_mm", 0))))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(tool.get("edges", 0))))
            self.table.setItem(row_index, 4, QTableWidgetItem(tool.get("condition", "")))
            self.table.setItem(row_index, 5, QTableWidgetItem(tool.get("notes", "")))
            
            tool_id = tool.get("tool_id")
            svg_exists = any(f.startswith(tool_id) and f.endswith("_area_vs_angle_plot.svg") for f in os.listdir(DATA_FOLDER_PATH))
            
            if svg_exists:
                btn = QPushButton("View Profile")
                btn.setObjectName("ActionButtonSuccess") # Use object names for styling
                btn.clicked.connect(lambda _, t=tool_id: self.view_profile(t))
                self.table.setCellWidget(row_index, 6, btn)
            else:
                btn = QPushButton("Data Missing")
                btn.setObjectName("ActionButtonFailure")
                btn.setEnabled(False)
                self.table.setCellWidget(row_index, 6, btn)

    def save_metadata_from_table(self):
        updated_data = []
        for row in range(self.table.rowCount()):
            updated_data.append({
                "tool_id": self.table.item(row, 0).text(),
                "type": self.table.item(row, 1).text(),
                "diameter_mm": int(self.table.item(row, 2).text()),
                "edges": int(self.table.item(row, 3).text()),
                "condition": self.table.item(row, 4).text(),
                "notes": self.table.item(row, 5).text()
            })
        
        success, message = self.metadata_manager.save(updated_data)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

    def view_profile(self, tool_id):
        QMessageBox.information(self, "Navigate", f"Switching to profile view for {tool_id}.")