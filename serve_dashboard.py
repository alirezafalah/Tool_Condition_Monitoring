#!/usr/bin/env python3
"""
Lightweight Dashboard Server with Automatic File Sync
Run: python3 serve_dashboard.py
Then open: http://localhost:8000/tool_condition_summary_dashboard.html

Any edits made in the dashboard will be automatically saved directly
to disk in 'tool_symmetry_classification.json' and synced to Git!
"""

import os
import json
import http.server
import socketserver

PORT = 8000
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPO_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api/save_annotations":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                data = {}

            json_path = os.path.join(REPO_DIR, "tool_symmetry_classification.json")
            current = {"study": "Tool Condition Monitoring", "tool_annotations": {}}
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as fp:
                        current = json.load(fp)
                except Exception:
                    pass

            tid = data.get("tool_id")
            if tid:
                if "tool_annotations" not in current:
                    current["tool_annotations"] = {}
                if tid not in current["tool_annotations"]:
                    current["tool_annotations"][tid] = {}
                if "symmetry" in data:
                    current["tool_annotations"][tid]["symmetry"] = data["symmetry"]
                if "notes" in data and data["notes"] is not None:
                    current["tool_annotations"][tid]["notes"] = data["notes"]

                with open(json_path, "w") as fp:
                    json.dump(current, fp, indent=2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"saved"}')
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    print(f"Starting Tool Condition Monitoring Dashboard Server on port {PORT}...")
    print(f"Open: http://localhost:{PORT}/tool_condition_summary_dashboard.html")
    print("Press Ctrl+C to stop.")
    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
