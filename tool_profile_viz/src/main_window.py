import os
import re
import copy
from PyQt6.QtWidgets import (QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, 
                             QWidget, QPushButton, QMessageBox, QHeaderView, QFileDialog, 
                             QHBoxLayout, QLabel, QSlider, QSpacerItem, QSizePolicy, QComboBox)
# --- NEW: Import QTimer for the visual effect ---
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from .metadata_manager import MetadataManager, DEFAULT_METADATA_PATH, DATA_ROOT
from .profile_window import ProfileWindow

TOOLS_PATH = os.path.join(DATA_ROOT, "tools")
PROFILES_PATH = os.path.join(DATA_ROOT, "1d_profiles")
BLURRED_PATH = os.path.join(DATA_ROOT, "blurred")
MASKS_PATH = os.path.join(DATA_ROOT, "masks")

STATUS_OPTIONS = ["","Finished", "Question", "Reinspect"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tool Profile Visualization")
        self.setGeometry(100, 100, 1200, 800)

        self.metadata_manager = MetadataManager()
        self.current_metadata_path = DEFAULT_METADATA_PATH
        self.base_font_size = self.font().pointSize()
        
        self.undo_stack = []
        self.redo_stack = []
        self.last_saved_state = copy.deepcopy(self.metadata_manager.get_all_tools())
        self._is_populating = False
        
        # --- FIX: Initialize the list to hold open windows ---
        self.open_windows = []

        controls_layout = QHBoxLayout()
        self._create_controls(controls_layout)
        
        self.table = QTableWidget()
        self.table.itemChanged.connect(self._on_item_changed)
        self._configure_table()
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.table)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._create_edit_actions()
        self.populate_table()

    # ... (all methods until populate_table are unchanged)
    def _create_controls(self, layout):
        load_button = QPushButton("Load Metadata")
        load_button.setObjectName("TopControlButton")
        load_button.clicked.connect(self.load_custom_json)
        
        self.save_button = QPushButton("Save Metadata")
        self.save_button.setObjectName("TopControlButton")
        self.save_button.clicked.connect(self.save_metadata_from_table)
        
        add_tool_button = QPushButton("Add New Tool")
        add_tool_button.setObjectName("TopControlButton")
        add_tool_button.clicked.connect(self.add_new_tool)
        
        layout.addWidget(load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(add_tool_button)
        
        reminder_label = QLabel("Make sure to save Metadata after changing any data.")
        layout.addWidget(reminder_label)

        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        zoom_label = QLabel("Table Zoom:")
        zoom_slider = QSlider(Qt.Orientation.Horizontal)
        zoom_slider.setRange(80, 200)
        zoom_slider.setValue(100)
        zoom_slider.setFixedWidth(200)
        zoom_slider.valueChanged.connect(self.change_zoom)
        
        layout.addWidget(zoom_label)
        layout.addWidget(zoom_slider)

    def _configure_table(self):
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels(["Tool ID", "Type", "Diameter (mm)", "Edges", "Condition", "Material", "Coating", "Background", "Color", "Notes", "Status", "Profile", "Actions"])
        header = self.table.horizontalHeader()
        # header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)

    def _create_edit_actions(self):
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._undo)
        self.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._redo)
        self.addAction(redo_action)

    def _push_state_to_undo_stack(self):
        self.undo_stack.append(copy.deepcopy(self.metadata_manager.get_all_tools()))
        self.redo_stack.clear()

    def _check_save_state(self):
        current_data = self._get_data_from_table()
        is_dirty = (current_data != self.last_saved_state)
        
        self.save_button.setObjectName("TopControlButtonDirty" if is_dirty else "TopControlButton")
        self.style().unpolish(self.save_button)
        self.style().polish(self.save_button)

    def _on_item_changed(self, item):
        if not self._is_populating:
            old_state = copy.deepcopy(self.metadata_manager.get_all_tools())
            new_state = self._get_data_from_table()
            self.undo_stack.append(old_state)
            self.redo_stack.clear()
            self.metadata_manager.metadata = new_state
            self._check_save_state()

    def _undo(self):
        if self.undo_stack:
            self.redo_stack.append(copy.deepcopy(self.metadata_manager.get_all_tools()))
            self.metadata_manager.metadata = self.undo_stack.pop()
            self.populate_table()

    def _redo(self):
        if self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.metadata_manager.get_all_tools()))
            self.metadata_manager.metadata = self.redo_stack.pop()
            self._check_save_state() 
            self.populate_table()

    def populate_table(self):
        self._is_populating = True
        tool_data = self.metadata_manager.get_all_tools()
        self.table.setRowCount(len(tool_data))

        # Check for data folders once
        if not os.path.isdir(PROFILES_PATH):
             QMessageBox.warning(self, "Warning", f"Could not find 1D profiles folder. 'View Profile' will be disabled.\nPath: {PROFILES_PATH}")
        if not os.path.isdir(BLURRED_PATH) or not os.path.isdir(MASKS_PATH):
             QMessageBox.warning(self, "Warning", f"Could not find image folders ('blurred', 'masks'). 'View Profile' may fail.\nPath: {DATA_ROOT}")
        
        for row, tool in enumerate(tool_data):
            self.table.setItem(row, 0, QTableWidgetItem(tool.get("tool_id", "")))
            self.table.setItem(row, 1, QTableWidgetItem(tool.get("type", "")))
            self.table.setItem(row, 2, QTableWidgetItem(str(tool.get("diameter_mm", 0))))
            self.table.setItem(row, 3, QTableWidgetItem(str(tool.get("edges", 0))))
            self.table.setItem(row, 4, QTableWidgetItem(tool.get("condition", "")))
            self.table.setItem(row, 5, QTableWidgetItem(tool.get("material", "")))
            self.table.setItem(row, 6, QTableWidgetItem(tool.get("coating", "")))
            self.table.setItem(row, 7, QTableWidgetItem(tool.get("background_type", "")))
            self.table.setItem(row, 8, QTableWidgetItem(tool.get("color", "")))
            self.table.setItem(row, 9, QTableWidgetItem(tool.get("notes", "")))
            status_combo = QComboBox()
            status_combo.addItems(STATUS_OPTIONS)
            current_status = tool.get("inspection_status", "")
            if current_status not in STATUS_OPTIONS:
                current_status = ""
            status_combo.setCurrentText(current_status)
            status_combo.currentTextChanged.connect(self._on_combo_changed)
            self.table.setCellWidget(row, 10, status_combo)

            tool_id = tool.get("tool_id")
            svg_file_path = os.path.join(PROFILES_PATH, f"{tool_id}_area_vs_angle_plot.svg")
            svg_exists = os.path.exists(svg_file_path)
            
            profile_btn = QPushButton("View Profile" if svg_exists else "Data Missing")
            profile_btn.setObjectName("ActionButtonSuccess" if svg_exists else "ActionButtonFailure")
            profile_btn.setEnabled(svg_exists)
            
            # --- ENHANCEMENT: Pass the button itself to the handler ---
            profile_btn.clicked.connect(lambda _, b=profile_btn, t=tool_id: self.view_profile(b, t))
            self.table.setCellWidget(row, 11, profile_btn)

            delete_btn = QPushButton("Delete")
            delete_btn.setObjectName("DeleteButton")
            delete_btn.clicked.connect(lambda _, r=row: self.delete_tool(r))
            self.table.setCellWidget(row, 12, delete_btn)
        self._is_populating = False
        self.table.resizeColumnsToContents()
        self._check_save_state()

    def _find_tool_files(self, tool_id):
        """Helper to find all necessary file and folder paths for a tool."""
        # --- UPDATED: Build paths directly using the new constants ---
        svg_path = os.path.join(PROFILES_PATH, f"{tool_id}_area_vs_angle_plot.svg")
        blurred_folder = os.path.join(BLURRED_PATH, f"{tool_id}_blurred")
        mask_folder = os.path.join(MASKS_PATH, f"{tool_id}_final_masks")
        tools_folder = os.path.join(TOOLS_PATH, tool_id)

        if not os.path.isdir(tools_folder):
            tools_folder = None
        if not os.path.exists(svg_path):
            svg_path = None
        if not os.path.isdir(blurred_folder):
            blurred_folder = None
        if not os.path.isdir(mask_folder):
            mask_folder = None

        # Find the 4 overview images from the blurred folder
        overview_paths = []
        if tools_folder:
            image_files = [f for f in os.listdir(tools_folder) if f.endswith(".tiff")]
            if image_files:
                degrees_files = {}
                for f in image_files:
                    match = re.search(r"(\d{4}\.\d{2})", f)
                    if match:
                        degrees_files[float(match.group(1))] = f
                
                if degrees_files:
                    for target_deg in [0, 90, 180, 270]:
                        closest_deg = min(degrees_files.keys(), key=lambda d: abs(d - target_deg))
                        overview_paths.append(os.path.join(tools_folder, degrees_files[closest_deg]))

        return svg_path, overview_paths, blurred_folder, mask_folder, tools_folder

    def view_profile(self, button, tool_id):
        """Finds tool files and launches the ProfileWindow with all necessary paths."""
        original_text = button.text()
        button.setText("Loading...")
        button.setEnabled(False)

        # Use the helper to find all required files and folders
        svg_path, overview_paths, blurred_folder, mask_folder, tools_folder = self._find_tool_files(tool_id)

        # A robust check to ensure all necessary paths were found
        if not all([svg_path, overview_paths, blurred_folder, mask_folder, tools_folder]):
            QMessageBox.warning(self, "Error", f"Missing some data folders or files for {tool_id}.")
            self._reset_button_state(button, original_text)
            return

        # Pass all the required paths to the new window's constructor
        win = ProfileWindow(tool_id, svg_path, overview_paths, blurred_folder, mask_folder, tools_folder)
        self.open_windows.append(win)
        
        QTimer.singleShot(400, lambda: self._reset_button_state(button, original_text))


    def _reset_button_state(self, button, text):
        """Helper function to restore the button's appearance."""
        button.setText(text)
        button.setEnabled(True)

    # ... (all methods from delete_tool onwards are unchanged)
    def delete_tool(self, row_index):
        tool_id = self.table.item(row_index, 0).text()
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                                     f"Are you sure you want to delete '{tool_id}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._push_state_to_undo_stack()
            scroll_pos = self.table.verticalScrollBar().value()
            self.metadata_manager.get_all_tools().pop(row_index)
            self.populate_table()
            self.table.verticalScrollBar().setValue(scroll_pos)

    def add_new_tool(self):
        self._push_state_to_undo_stack()
        tool_data = self.metadata_manager.get_all_tools()
        max_num = 0
        for tool in tool_data:
            match = re.search(r'\d+', tool.get("tool_id", ""))
            if match: max_num = max(max_num, int(match.group(0)))
        new_id = f"tool{max_num + 1:03d}"
        new_tool = {
            "tool_id": new_id, "type": "", "diameter_mm": 0, "edges": 0,
            "condition": "Unknown", "material": "", "coating": "", "background_type": "", "notes": "", "color": "", "inspection_status": ""
        }
        tool_data.append(new_tool)
        self.populate_table()
        self.table.scrollToBottom()

    def _get_data_from_table(self):
        data = []
        for row in range(self.table.rowCount()):
            def safe_int(item_text):
                try:
                    return int(item_text)
                except (ValueError, TypeError):
                    return 0
            status_widget = self.table.cellWidget(row, 10)
            status_text = status_widget.currentText() if status_widget else ""
            data.append({
                "tool_id": self.table.item(row, 0).text(), 
                "type": self.table.item(row, 1).text(),
                "diameter_mm": safe_int(self.table.item(row, 2).text()), 
                "edges": safe_int(self.table.item(row, 3).text()),
                "condition": self.table.item(row, 4).text(), 
                "material": self.table.item(row, 5).text(),
                "coating": self.table.item(row, 6).text(),
                "background_type": self.table.item(row, 7).text(), 
                "color": self.table.item(row, 8).text(),
                "notes": self.table.item(row, 9).text(),
                "inspection_status": status_text,
            })
        return data

    def save_metadata_from_table(self):
        updated_data = self._get_data_from_table()
        success, message = self.metadata_manager.save(self.current_metadata_path, updated_data)
        if success:
            self.last_saved_state = copy.deepcopy(updated_data)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._check_save_state()
        QMessageBox.information(self, "Status", message) if success else QMessageBox.critical(self, "Error", message)

    def load_custom_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Metadata File", "", "JSON Files (*.json)")
        if filepath:
            if self.metadata_manager.load(filepath):
                self.current_metadata_path = filepath
                self.last_saved_state = copy.deepcopy(self.metadata_manager.get_all_tools())
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.populate_table()
                self.setWindowTitle(f"Tool Profile Visualization - {os.path.basename(filepath)}")
                QMessageBox.information(self, "Success", f"Loaded {os.path.basename(filepath)}")
            else:
                QMessageBox.critical(self, "Error", f"Could not load or parse the file: {filepath}")

    def change_zoom(self, value):
        font = self.table.font()
        new_size = int(self.base_font_size * (value / 100.0))
        font.setPointSize(new_size)
        self.table.setFont(font)

    def _on_combo_changed(self, text):
        """Called when a status dropdown is changed by the user."""
        if not self._is_populating:
            # Save undo state
            old_state = copy.deepcopy(self.metadata_manager.get_all_tools())
            new_state = self._get_data_from_table()
            self.undo_stack.append(old_state)
            self.redo_stack.clear()
            self.metadata_manager.metadata = new_state
            self._check_save_state()