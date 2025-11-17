import os
import json
import csv

# The default path is still inside the app directory
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_METADATA_PATH = os.path.join(APP_ROOT, 'tools_metadata.json')

# The DATA_ROOT is one level up from Tool_Condition_Monitoring, in the 'DATA' folder
DATA_ROOT = os.path.abspath(os.path.join(APP_ROOT, "..", "..", "DATA"))

class MetadataManager:
    def __init__(self):
        # Load the default data on startup
        self.metadata = self.load(DEFAULT_METADATA_PATH)
        if not self.metadata: # If loading failed or file was missing
            self.metadata = self._create_default()
            self.save(DEFAULT_METADATA_PATH, self.metadata)

    def _create_default(self):
            """Creates the default data structure for the tools."""
            default_data = []
            for i in range(1, 117):
                tool_id = f"tool{i:03d}"
                default_data.append({
                    "tool_id": tool_id, 
                    "type": "", 
                    "diameter_mm": 0, 
                    "edges": 0,
                    "condition": "Unknown", 
                    "material": "",
                    "coating": "",
                    "background_type": "",
                    "notes": "",
                    "color": "",
                    "inspection_status": ""
                })
            return default_data

    def load(self, filepath):
        """Loads metadata from a specific JSON file path."""
        if not os.path.exists(filepath):
            return None # Return None if the file doesn't exist
        try:
            with open(filepath, 'r') as f:
                self.metadata = json.load(f)
                return self.metadata
        except (json.JSONDecodeError, IOError):
            return None # Return None on error

    def get_all_tools(self):
        """Returns the currently loaded list of all tool data."""
        return self.metadata

    # --- UPDATED: save() method ---
    def save(self, filepath, data):
        """
        Saves the provided data structure to:
        1. The JSON file (at 'filepath')
        2. The CSV file (in the DATA_ROOT folder)
        """

        csv_filename = os.path.splitext(os.path.basename(filepath))[0] + ".csv"
        csv_filepath = os.path.join(DATA_ROOT, csv_filename)

        json_success = False
        json_message = ""

        # --- Step 1: Save the JSON (critical for the app) ---
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            json_success = True
            json_message = "Metadata saved successfully"
        except Exception as e:
            return False, f"Failed to save JSON metadata: {e}"

        # --- Step 2: Save the CSV (for your supervisor) ---
        if not data: 
            return True, "Metadata saved (JSON), but no data to save to CSV."

        try:
            headers = data[0].keys()
            
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            
            return True, f"{json_message}."

        except Exception as e:
            # Inform the user, but this is not a critical failure
            return True, f"{json_message}, but failed to save CSV: {e}"