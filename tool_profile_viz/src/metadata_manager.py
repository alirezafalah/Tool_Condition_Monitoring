import os
import json

# Correctly define the path to the metadata file.
# It should be inside the 'tool_profile_viz' folder.
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This is tool_profile_viz/
METADATA_PATH = os.path.join(APP_ROOT, 'tools_metadata.json')

class MetadataManager:
    def __init__(self):
        self.metadata = self.load_or_create()

    def load_or_create(self):
        """Loads metadata from JSON or creates a default file if it doesn't exist."""
        if not os.path.exists(METADATA_PATH):
            default_data = []
            for i in range(1, 117):
                tool_id = f"tool{i:03d}"
                default_data.append({
                    "tool_id": tool_id, "type": "", "diameter_mm": 0, "edges": 0,
                    "condition": "Unknown", "notes": ""
                })
            self.save(default_data)
            return default_data
        else:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)

    def get_all_tools(self):
        """Returns the list of all tool data."""
        return self.metadata

    def save(self, data):
        """Saves the provided data structure to the JSON file."""
        try:
            with open(METADATA_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            return True, "Metadata saved successfully!"
        except Exception as e:
            return False, f"Failed to save metadata: {e}"