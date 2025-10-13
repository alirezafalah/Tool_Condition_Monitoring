import os
import re
from PyQt6.QtWidgets import (QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, 
                             QWidget, QPushButton, QMessageBox, QHeaderView, QFileDialog, 
                             QHBoxLayout, QLabel, QSlider, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt
from .metadata_manager import MetadataManager, DEFAULT_METADATA_PATH

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tool Profile Visualization")
        self.setGeometry(100, 100, 1200, 800)

        self.metadata_manager = MetadataManager()
        self.current_metadata_path = DEFAULT_METADATA_PATH
        self.base_font_size = self.font().pointSize()

        # --- Main Layout ---
        main_layout = QVBoxLayout()

        # --- Top Controls Layout (Buttons on left, Zoom on right) ---
        controls_layout = QHBoxLayout()

        # Action Buttons (formerly in File menu)
        load_button = QPushButton("Load Metadata")
        load_button.setObjectName("TopControlButton")
        load_button.clicked.connect(self.load_custom_json)
        
        save_button = QPushButton("Save Metadata")
        save_button.setObjectName("TopControlButton")
        save_button.clicked.connect(self.save_metadata_from_table)
        
        add_tool_button = QPushButton("Add New Tool")
        add_tool_button.setObjectName("TopControlButton")
        add_tool_button.clicked.connect(self.add_new_tool)
        
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(save_button)
        controls_layout.addWidget(add_tool_button)
        
        # Spacer pushes the zoom controls to the far right
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        controls_layout.addSpacerItem(spacer)

        # Zoom Controls
        controls_label = QLabel("Table Zoom:")
        zoom_slider = QSlider(Qt.Orientation.Horizontal)
        zoom_slider.setRange(80, 200)
        zoom_slider.setValue(100)
        zoom_slider.setFixedWidth(200)
        zoom_slider.valueChanged.connect(self.change_zoom)
        
        controls_layout.addWidget(controls_label)
        controls_layout.addWidget(zoom_slider)
        main_layout.addLayout(controls_layout)

        # --- Data Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Tool ID", "Type", "Diameter (mm)", "Edges", "Condition", "Notes", "Profile", "Actions"])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.populate_table()
        main_layout.addWidget(self.table)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def populate_table(self):
        tool_data = self.metadata_manager.get_all_tools()
        self.table.setRowCount(len(tool_data))
        for row, tool in enumerate(tool_data):
            self.table.setItem(row, 0, QTableWidgetItem(tool.get("tool_id", "")))
            self.table.setItem(row, 1, QTableWidgetItem(tool.get("type", "")))
            self.table.setItem(row, 2, QTableWidgetItem(str(tool.get("diameter_mm", 0))))
            self.table.setItem(row, 3, QTableWidgetItem(str(tool.get("edges", 0))))
            self.table.setItem(row, 4, QTableWidgetItem(tool.get("condition", "")))
            self.table.setItem(row, 5, QTableWidgetItem(tool.get("notes", "")))
            
            tool_id = tool.get("tool_id")
            svg_exists = any(f.startswith(tool_id) and f.endswith("_area_vs_angle_plot.svg") for f in os.listdir(DATA_FOLDER_PATH))
            
            profile_btn = QPushButton("View Profile" if svg_exists else "Data Missing")
            profile_btn.setObjectName("ActionButtonSuccess" if svg_exists else "ActionButtonFailure")
            profile_btn.setEnabled(svg_exists)
            profile_btn.clicked.connect(lambda _, t=tool_id: self.view_profile(t))
            self.table.setCellWidget(row, 6, profile_btn)

            delete_btn = QPushButton("Delete")
            delete_btn.setObjectName("DeleteButton")
            delete_btn.clicked.connect(lambda _, r=row: self.delete_tool(r))
            self.table.setCellWidget(row, 7, delete_btn)

    def delete_tool(self, row_index):
        """Deletes a tool, preserving the scroll position."""
        tool_id = self.table.item(row_index, 0).text()
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                                     f"Are you sure you want to delete '{tool_id}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # --- FIX: Preserve scroll position ---
            scroll_position = self.table.verticalScrollBar().value()
            
            self.metadata_manager.get_all_tools().pop(row_index)
            self.populate_table()
            
            self.table.verticalScrollBar().setValue(scroll_position)

    def load_custom_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Metadata File", "", "JSON Files (*.json)")
        if filepath:
            if self.metadata_manager.load(filepath):
                self.current_metadata_path = filepath
                self.populate_table()
                self.setWindowTitle(f"Tool Profile Visualization - {os.path.basename(filepath)}")
                QMessageBox.information(self, "Success", f"Loaded {os.path.basename(filepath)}")
            else:
                QMessageBox.critical(self, "Error", f"Could not load or parse the file: {filepath}")

    def add_new_tool(self):
        tool_data = self.metadata_manager.get_all_tools()
        max_num = 0
        for tool in tool_data:
            match = re.search(r'\d+', tool.get("tool_id", ""))
            if match:
                max_num = max(max_num, int(match.group(0)))
        
        new_id = f"tool{max_num + 1:03d}"
        new_tool = {"tool_id": new_id, "type": "", "diameter_mm": 0, "edges": 0, "condition": "Unknown", "notes": ""}
        tool_data.append(new_tool)
        self.populate_table()

    def save_metadata_from_table(self):
        updated_data = []
        for row in range(self.table.rowCount()):
            updated_data.append({
                "tool_id": self.table.item(row, 0).text(), "type": self.table.item(row, 1).text(),
                "diameter_mm": int(self.table.item(row, 2).text()), "edges": int(self.table.item(row, 3).text()),
                "condition": self.table.item(row, 4).text(), "notes": self.table.item(row, 5).text()
            })
        
        success, message = self.metadata_manager.save(self.current_metadata_path, updated_data)
        QMessageBox.information(self, "Status", message) if success else QMessageBox.critical(self, "Error", message)

    def change_zoom(self, value):
        font = self.table.font()
        new_size = int(self.base_font_size * (value / 100.0))
        font.setPointSize(new_size)
        self.table.setFont(font)

    def view_profile(self, tool_id):
        QMessageBox.information(self, "Navigate", f"Switching to profile view for {tool_id}.")