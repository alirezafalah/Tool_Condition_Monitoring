import os
import json

# The default path is still inside the app directory
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_METADATA_PATH = os.path.join(APP_ROOT, 'tools_metadata.json')

class MetadataManager:
    def __init__(self):
        # Load the default data on startup
        self.metadata = self.load(DEFAULT_METADATA_PATH)
        if not self.metadata: # If loading failed or file was missing
            self.metadata = self._create_default()
            self.save(DEFAULT_METADATA_PATH, self.metadata)

    def _create_default(self):
        """Creates the default data structure for 116 tools."""
        default_data = []
        for i in range(1, 117):
            tool_id = f"tool{i:03d}"
            default_data.append({
                "tool_id": tool_id, "type": "", "diameter_mm": 0, "edges": 0,
                "condition": "Unknown", "notes": ""
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

    def save(self, filepath, data):
        """Saves the provided data structure to a specific JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True, "Metadata saved successfully!"
        except Exception as e:
            return False, f"Failed to save metadata: {e}"