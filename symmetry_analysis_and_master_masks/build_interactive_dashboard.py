#!/usr/bin/env python3
"""
Generate an Interactive HTML Dashboard & Comprehensive Markdown/CSV Inventory
for all 115 rotary cutting tools in the Condition Monitoring Study.

Key Features & Updates:
- NO DEFAULTS for symmetry classification: all tools start as Unassigned ("⚪ — Select —").
- Striking COLOR CODING:
  - Symmetrical: 🟢 Rich Emerald Green (#064e3b bg, #4ade80 text, #22c55e border, glowing shadow)
  - Asymmetrical: 🔴 Deep Crimson Red (#7f1d1d bg, #fca5a5 text, #ef4444 border, glowing shadow)
  - Unassigned: ⚪ Neutral Slate (#1e293b bg, #94a3b8 text, #475569 border)
  - Options styled with explicit background colors for Linux GTK/browser dropdown menus
  - Row accent border tint when classified
- PERSISTENCE FIX:
  - Initialization checks `document.readyState` so `loadStoredAnnotations()` runs immediately on load!
  - `Ctrl+S` / `Cmd+S` keyboard shortcut and "💾 Save HTML (Ctrl+S)" button to bake user selections directly into HTML file!
  - In-place row editing NEVER triggers table filtering or hides rows under the user's cursor.
- Live KPI counter tracking: Symmetrical, Asymmetrical, and Pending Classification.
- Exact requested column order:
  Tool ID | Status | Symmetrical | Mean DSI | Max DSI | Zone | View | Type | Flute Number | Org Condition | Work Condition | Physical Notes & Edits | Dia | Max Diff | ROI Height
- "Export JSON" generating explicit lists of symmetrical, asymmetrical, and unassigned tools.
- "Export CSV" dynamically capturing active selections and edited notes.
- High-res modal preview + new tab view for every tool.
"""

import os
import sys
import json
import shutil
import pandas as pd
import numpy as np

def generate_dashboard():
    base_dir = "/home/alifalah/Projects/DATA"
    repo_dir = "/home/alifalah/Projects/Tool_Condition_Monitoring"
    reports_dir = os.path.join(base_dir, "reports")
    max_dsi_dir = os.path.join(base_dir, "max_dsi_tools")
    pixel_dir = os.path.join(base_dir, "pixel_comparison_per_tool")
    artifact_dir = "/home/alifalah/.gemini/antigravity-cli/brain/9d13cb4f-0909-42c3-861a-1312e1a01d3c"

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(max_dsi_dir, exist_ok=True)

    # 1. Load data
    class_csv = os.path.join(max_dsi_dir, "full_population_max_dsi_classification.csv")
    meta_csv = os.path.join(base_dir, "tools_metadata.csv")

    df_class = pd.read_csv(class_csv)
    df_meta = pd.read_csv(meta_csv)
    meta_dict = {r["tool_id"]: r.to_dict() for _, r in df_meta.iterrows()}

    # 2. Enrich tool records
    rows = []
    for _, r in df_class.iterrows():
        tid = r["tool_id"]
        meta_item = meta_dict.get(tid, {})

        is_kept = r["role"] != "excluded"
        status_label = "Kept (Active)" if is_kept else "Discarded (Excluded)"

        # Human-readable exclusion reason
        excl_reason = r["exclusion_reason"]
        if excl_reason == "EXCLUDED_TOPPER":
            reason_clean = "Threading Tap (Inherent Helical/Pitch Asymmetry)"
        elif excl_reason == "EXCLUDED_ASYMMETRIC_TOOL":
            reason_clean = "Asymmetric Geometry / Non-Symmetric Breakdown"
        elif excl_reason == "EXCLUDED_SEGMENTATION_ARTIFACT":
            reason_clean = "Camera Framing / Optical Backlight Artifact"
        elif excl_reason == "EXCLUDED_CUSTOM_SCREENING":
            reason_clean = "User-Excluded Tool (Screened Outlier / Asymmetry)"
        else:
            reason_clean = "Kept in Active Dataset"

        # 3-Zone label
        zone_raw = r["zone_redef_1p5_4p0"]
        if not is_kept:
            zone_clean = "Excluded"
            zone_badge = "badge-excluded"
        elif zone_raw == "ZONE_1_PASS":
            zone_clean = "Zone 1: Pass (≤ 1.5%)"
            zone_badge = "badge-zone1"
        elif zone_raw == "ZONE_2_MANUAL":
            zone_clean = "Zone 2: Manual (1.5% - 4.0%)"
            zone_badge = "badge-zone2"
        elif zone_raw == "ZONE_3_REJECT":
            zone_clean = "Zone 3: Reject (> 4.0%)"
            zone_badge = "badge-zone3"
        else:
            zone_clean = "Unknown"
            zone_badge = "badge-neutral"

        # Working condition
        working_cond = r["group"]
        orig_cond = str(r["condition_raw"]).strip().lower()

        # ROI height
        roi_h = 200
        sym_meta_path = os.path.join(pixel_dir, f"{tid}_symmetry_metadata.json")
        if os.path.exists(sym_meta_path):
            with open(sym_meta_path, "r") as fp:
                sm = json.load(fp)
            roi_h = sm.get("dynamic_roi", {}).get("roi_height_px", np.nan)
        if tid == "tool073":
            roi_h = 400
        elif tid == "tool092":
            roi_h = 180

        # Plot relative paths
        # From /home/alifalah/Projects/Tool_Condition_Monitoring/
        img_rel_repo = f"../DATA/pixel_comparison_per_tool/{tid}_overlay_dsi_stacked.png"
        # From /home/alifalah/Projects/DATA/reports/
        img_rel_reports = f"../pixel_comparison_per_tool/{tid}_overlay_dsi_stacked.png"
        img_exists = os.path.exists(os.path.join(pixel_dir, f"{tid}_overlay_dsi_stacked.png"))

        # Physical notes / highlights
        notes = str(r["notes"]) if pd.notna(r["notes"]) and str(r["notes"]).strip().lower() != "nan" else ""
        highlights = []
        if tid == "tool003":
            highlights.append("Wavy roughing knuckles (sinusoidal profile, excluded as asymmetric)")
        elif tid == "tool006":
            highlights.append("Severe Flute 1 corner radius breakdown (fractured cutting edge)")
        elif tid == "tool008":
            highlights.append("4mm stepped cutter with 6mm shank (excluded as asymmetric)")
        elif tid == "tool073":
            highlights.append("Custom 400px ROI (captures full cutting lips, Peak DSI 1.08%)")
        elif tid == "tool092":
            highlights.append("Custom 180px ROI (isolates cutting tip micro-chipping, Peak DSI 1.60%)")
        elif tid in ["tool057", "tool076"]:
            highlights.append("Reclassified from minor uniform wear to used")
        elif excl_reason == "EXCLUDED_TOPPER":
            highlights.append("Excluded: Thread tap helical pitch lead")
        elif not is_kept:
            highlights.append(f"Excluded: {reason_clean}")

        if notes:
            highlights.append(notes)

        full_notes = " | ".join(highlights) if highlights else "Standard symmetric geometry"

        rows.append({
            "tool_id": tid,
            "status": status_label,
            "is_kept": is_kept,
            "symmetry": "Unassigned",  # Clean baseline: user fills gradually
            "exclusion_reason": reason_clean,
            "original_condition": orig_cond,
            "working_condition": working_cond,
            "tool_type": r["type"],
            "diameter_mm": r["diameter_mm"],
            "edges": int(r["edges"]),
            "mean_dsi": r["max_pair_mean_dsi"] if pd.notna(r["max_pair_mean_dsi"]) else np.nan,
            "peak_dsi": r["max_peak_dsi"] if pd.notna(r["max_peak_dsi"]) else np.nan,
            "overall_dsi": r["overall_dsi"] if pd.notna(r["overall_dsi"]) else np.nan,
            "max_pixel_diff": r["max_abs_diff_pixels"] if pd.notna(r["max_abs_diff_pixels"]) else np.nan,
            "roi_height_px": int(roi_h) if pd.notna(roi_h) else np.nan,
            "zone_clean": zone_clean,
            "zone_badge": zone_badge,
            "notes": full_notes,
            "img_rel_repo": img_rel_repo,
            "img_rel_reports": img_rel_reports,
            "img_exists": img_exists
        })

    full_df = pd.DataFrame(rows)

    # 3. Save Summary CSV & Markdown in exact requested ordering
    csv_cols = [
        "tool_id", "status", "symmetry", "mean_dsi", "peak_dsi", "zone_clean",
        "tool_type", "edges", "original_condition", "working_condition",
        "notes", "diameter_mm", "max_pixel_diff", "roi_height_px"
    ]
    csv_out = os.path.join(reports_dir, "full_tool_inventory_and_dsi_summary.csv")
    full_df[csv_cols].to_csv(csv_out, index=False)
    shutil.copy2(csv_out, os.path.join(max_dsi_dir, "full_tool_inventory_and_dsi_summary.csv"))
    shutil.copy2(csv_out, os.path.join(repo_dir, "full_tool_inventory_and_dsi_summary.csv"))

    # Generate Markdown Table Document in requested column ordering
    md_lines = [
        "# Complete 115-Tool Inventory, DSI Metrics & Condition Triage Summary",
        "",
        "| Tool ID | Status | Symmetry | Mean DSI (%) | Max DSI (%) | Zone | Type | Flutes | Org Cond | Work Cond | Physical & User Notes | Dia (mm) | Max Diff (px) | ROI (px) |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |"
    ]
    for _, r in full_df.iterrows():
        p_dsi = f"{r['peak_dsi']:.2f}%" if pd.notna(r["peak_dsi"]) else "N/A"
        m_dsi = f"{r['mean_dsi']:.2f}%" if pd.notna(r["mean_dsi"]) else "N/A"
        roi_s = f"{r['roi_height_px']}px" if pd.notna(roi_h) else "N/A"
        diff_s = f"{r['max_pixel_diff']:,.0f}" if pd.notna(r["max_pixel_diff"]) else "N/A"
        md_lines.append(
            f"| **`{r['tool_id']}`** | {r['status']} | {r['symmetry']} | {m_dsi} | **{p_dsi}** | "
            f"**{r['zone_clean']}** | {r['tool_type']} | {r['edges']} | {r['original_condition']} | **{r['working_condition']}** | "
            f"{r['notes']} | {r['diameter_mm']:.1f} | {diff_s} | {roi_s} |"
        )
    md_content = "\n".join(md_lines)
    md_out = os.path.join(reports_dir, "full_tool_inventory_and_dsi_summary.md")
    with open(md_out, "w") as fp:
        fp.write(md_content)
    shutil.copy2(md_out, os.path.join(max_dsi_dir, "full_tool_inventory_and_dsi_summary.md"))
    shutil.copy2(md_out, os.path.join(repo_dir, "full_tool_inventory_and_dsi_summary.md"))

    # 4. Compute KPIs for HTML
    total_tools = len(full_df)
    kept_tools = int((full_df["status"] == "Kept (Active)").sum())
    excl_tools = int((full_df["status"] == "Discarded (Excluded)").sum())
    z1_tools = int((full_df["zone_clean"] == "Zone 1: Pass (≤ 1.5%)").sum())
    z2_tools = int((full_df["zone_clean"] == "Zone 2: Manual (1.5% - 4.0%)").sum())
    z3_tools = int((full_df["zone_clean"] == "Zone 3: Reject (> 4.0%)").sum())

    # 5. Build HTML Generator Function
    def build_html(image_path_key):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Condition Monitoring — Full Tool Inventory & DSI Dashboard</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --primary-hover: #0284c7;
            --zone-1: #22c55e;
            --zone-2: #f59e0b;
            --zone-3: #ef4444;
            --excluded: #64748b;
            --cond-new: #10b981;
            --cond-used: #38bdf8;
            --cond-worn: #fb923c;
            --cond-deposit: #c084fc;
            --cond-fractured: #f43f5e;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1750px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}

        h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 4px;
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .btn-action {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.15s;
        }}

        .btn-save-html {{
            background-color: #10b981;
            color: #0f172a;
            border-color: #34d399;
        }}
        .btn-save-html:hover {{
            background-color: #059669;
            color: #ffffff;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.45);
        }}

        .btn-save-json {{
            background-color: #0284c7;
            color: #ffffff;
            border-color: #38bdf8;
        }}
        .btn-save-json:hover {{
            background-color: #0369a1;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
        }}

        .btn-export {{
            background-color: #334155;
            color: var(--text-main);
            border-color: var(--border-color);
        }}
        .btn-export:hover {{
            background-color: #475569;
            color: #ffffff;
        }}

        .btn-reset-data {{
            background-color: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border-color: rgba(239, 68, 68, 0.3);
        }}
        .btn-reset-data:hover {{
            background-color: rgba(239, 68, 68, 0.3);
        }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
            gap: 14px;
            margin-bottom: 22px;
        }}

        .kpi-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: transform 0.15s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
        }}

        .kpi-title {{
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 4px;
        }}

        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--text-main);
        }}

        .kpi-subtext {{
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 3px;
        }}

        .kpi-sym {{ border-left: 4px solid #4ade80; }}
        .kpi-asym {{ border-left: 4px solid #f87171; }}
        .kpi-pending {{ border-left: 4px solid #94a3b8; }}
        .kpi-z1 {{ border-left: 4px solid var(--zone-1); }}
        .kpi-z2 {{ border-left: 4px solid var(--zone-2); }}
        .kpi-z3 {{ border-left: 4px solid var(--zone-3); }}

        /* Filter Controls */
        .controls-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 18px;
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            align-items: center;
        }}

        .search-box {{
            flex: 1 1 240px;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 9px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.15s;
        }}

        .search-box input:focus {{
            border-color: var(--primary);
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .filter-group label {{
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            white-space: nowrap;
        }}

        select {{
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }}

        select:focus {{
            border-color: var(--primary);
        }}

        .btn-reset {{
            background-color: #334155;
            color: var(--text-main);
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }}

        .btn-reset:hover {{
            background-color: #475569;
        }}

        .counter-badge {{
            margin-left: auto;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-muted);
        }}

        /* Table Styling */
        .table-wrapper {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: auto;
            max-height: 72vh;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            text-align: left;
        }}

        thead {{
            position: sticky;
            top: 0;
            background-color: #1e293b;
            z-index: 10;
            box-shadow: 0 1px 0 var(--border-color);
        }}

        th {{
            padding: 11px 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.73rem;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            transition: color 0.15s;
        }}

        th:hover {{
            color: var(--primary);
        }}

        th .sort-icon {{
            margin-left: 4px;
            font-size: 0.68rem;
            opacity: 0.5;
        }}

        th.sorted-asc .sort-icon,
        th.sorted-desc .sort-icon {{
            opacity: 1;
            color: var(--primary);
        }}

        tbody tr {{
            border-bottom: 1px solid #1e293b;
            transition: background-color 0.15s;
        }}

        tbody tr:hover {{
            background-color: rgba(56, 189, 248, 0.05);
        }}

        tbody tr.row-excluded {{
            opacity: 0.75;
            background-color: rgba(15, 23, 42, 0.35);
        }}

        /* Subtle row left border accents when classified */
        tbody tr.row-sym {{
            border-left: 3px solid #22c55e;
        }}

        tbody tr.row-asym {{
            border-left: 3px solid #ef4444;
            background-color: rgba(239, 68, 68, 0.04);
        }}

        td {{
            padding: 10px 12px;
            vertical-align: middle;
            white-space: nowrap;
        }}

        .tool-id-cell {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-weight: 700;
            color: var(--primary);
        }}

        /* STRIKING COLOR-CODED SYMMETRY DROPDOWN */
        .symmetry-select {{
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.81rem;
            font-weight: 800;
            cursor: pointer;
            border: 2px solid var(--border-color);
            background-color: #0f172a;
            color: var(--text-muted);
            transition: all 0.2s ease;
            outline: none;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }}

        .symmetry-select.sym-unassigned {{
            background-color: #1e293b !important;
            color: #94a3b8 !important;
            border-color: #475569 !important;
        }}

        .symmetry-select.sym-symmetrical {{
            background-color: #064e3b !important;
            color: #4ade80 !important;
            border-color: #22c55e !important;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.35) !important;
        }}

        .symmetry-select.sym-asymmetrical {{
            background-color: #7f1d1d !important;
            color: #fca5a5 !important;
            border-color: #ef4444 !important;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.35) !important;
        }}

        .symmetry-select option {{
            padding: 8px;
            font-weight: 700;
        }}

        .symmetry-select option[value=""] {{
            background-color: #1e293b;
            color: #94a3b8;
        }}

        .symmetry-select option[value="Symmetrical"] {{
            background-color: #064e3b;
            color: #4ade80;
        }}

        .symmetry-select option[value="Asymmetrical"] {{
            background-color: #7f1d1d;
            color: #fca5a5;
        }}

        /* Editable Notes Styling */
        .editable-note {{
            min-height: 22px;
            max-width: 320px;
            white-space: normal;
            word-wrap: break-word;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px dashed transparent;
            color: var(--text-muted);
            font-size: 0.81rem;
            line-height: 1.35;
            transition: all 0.2s;
            outline: none;
            cursor: text;
        }}

        .editable-note:hover {{
            border-color: var(--border-color);
            background-color: rgba(255, 255, 255, 0.04);
            color: var(--text-main);
        }}

        .editable-note:focus {{
            border: 1px solid var(--primary);
            background-color: #0f172a;
            color: var(--text-main);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }}

        .editable-note:empty::before {{
            content: "Click to add note...";
            color: #475569;
            font-style: italic;
        }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }}

        .badge-zone1 {{ background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
        .badge-zone2 {{ background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-zone3 {{ background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-excluded {{ background-color: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }}

        .badge-new {{ background-color: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .badge-used {{ background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; }}
        .badge-worn {{ background-color: rgba(251, 146, 60, 0.2); color: #fb923c; }}
        .badge-deposit {{ background-color: rgba(192, 132, 252, 0.2); color: #c084fc; }}
        .badge-fractured {{ background-color: rgba(244, 63, 94, 0.2); color: #f43f5e; }}

        .badge-status-kept {{ background-color: rgba(2, 132, 199, 0.2); color: #38bdf8; }}
        .badge-status-excl {{ background-color: rgba(100, 116, 139, 0.2); color: #cbd5e1; }}

        .dsi-cell {{
            font-family: ui-monospace, monospace;
            font-weight: 700;
        }}

        .btn-inspect {{
            display: inline-block;
            background-color: #334155;
            color: var(--primary);
            text-decoration: none;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            transition: all 0.15s;
            border: 1px solid #475569;
            cursor: pointer;
        }}

        .btn-inspect:hover {{
            background-color: var(--primary);
            color: #0f172a;
            border-color: var(--primary);
        }}

        /* Toast Indicator */
        #toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background-color: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--primary);
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            display: none;
            z-index: 1000;
        }}

        /* Modal Preview */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.88);
            backdrop-filter: blur(4px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        }}

        .modal-content {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-width: 92vw;
            max-height: 92vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }}

        .modal-header {{
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .modal-header h3 {{
            font-size: 1rem;
            color: var(--text-main);
        }}

        .modal-body {{
            padding: 16px;
            overflow: auto;
            text-align: center;
        }}

        .modal-body img {{
            max-width: 100%;
            max-height: 75vh;
            border-radius: 6px;
        }}

        .modal-close {{
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 1.4rem;
            cursor: pointer;
            padding: 0 6px;
        }}
        .modal-close:hover {{
            color: var(--text-main);
        }}

        /* Footer */
        .footer-bar {{
            margin-top: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--text-muted);
            font-size: 0.84rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div>
                <h1>Tool Condition Monitoring & Symmetry Analysis Dashboard</h1>
                <div class="subtitle">115-Tool Study: Manual Symmetry Classification Workspace, Peak DSI, and 3-Zone Quality Control</div>
            </div>
            <div class="header-actions">
                <button class="btn-action btn-save-html" onclick="saveHtmlToDisk()" title="Save all selections into the HTML file for Git version control (Ctrl+S)">
                    💾 Save HTML (Ctrl+S)
                </button>
                <button class="btn-action btn-save-json" onclick="exportSymmetryJSON()" title="Download Symmetry Classification JSON">
                    📋 Export JSON
                </button>
                <button class="btn-action btn-export" onclick="exportTableToCSV('tool_condition_summary.csv')">
                    📊 Export CSV
                </button>
                <button class="btn-action btn-reset-data" onclick="resetUserEdits()" title="Clear all selections back to empty baseline">
                    ↺ Clear All
                </button>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Total Tool Population</div>
                <div class="kpi-value">{total_tools}</div>
                <div class="kpi-subtext">115 Tools Scanned</div>
            </div>
            <div class="kpi-card kpi-sym">
                <div class="kpi-title">Symmetrical Tools</div>
                <div class="kpi-value" id="kpiSymCount" style="color: #4ade80;">0</div>
                <div class="kpi-subtext">Classified Symmetrical</div>
            </div>
            <div class="kpi-card kpi-asym">
                <div class="kpi-title">Asymmetrical Tools</div>
                <div class="kpi-value" id="kpiAsymCount" style="color: #f87171;">0</div>
                <div class="kpi-subtext">Classified Asymmetrical</div>
            </div>
            <div class="kpi-card kpi-pending">
                <div class="kpi-title">Pending Classification</div>
                <div class="kpi-value" id="kpiPendingCount" style="color: #94a3b8;">115</div>
                <div class="kpi-subtext">Awaiting User Decision</div>
            </div>
            <div class="kpi-card kpi-z1">
                <div class="kpi-title">Zone 1: Automated Pass</div>
                <div class="kpi-value" style="color: #4ade80;">{z1_tools}</div>
                <div class="kpi-subtext">Peak DSI ≤ 1.50%</div>
            </div>
            <div class="kpi-card kpi-z2">
                <div class="kpi-title">Zone 2: Manual Review</div>
                <div class="kpi-value" style="color: #fbbf24;">{z2_tools}</div>
                <div class="kpi-subtext">1.50% &lt; Peak DSI ≤ 4.00%</div>
            </div>
            <div class="kpi-card kpi-z3">
                <div class="kpi-title">Zone 3: Automated Reject</div>
                <div class="kpi-value" style="color: #f87171;">{z3_tools}</div>
                <div class="kpi-subtext">Peak DSI &gt; 4.00%</div>
            </div>
        </div>

        <!-- Controls & Filters -->
        <div class="controls-card">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search tool ID, type, condition, notes..." oninput="filterTable()">
            </div>

            <div class="filter-group">
                <label for="statusFilter">Status:</label>
                <select id="statusFilter" onchange="filterTable()">
                    <option value="ALL">All Tools ({total_tools})</option>
                    <option value="Kept (Active)">Kept (Active) ({kept_tools})</option>
                    <option value="Discarded (Excluded)">Discarded (Excluded) ({excl_tools})</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="symFilter">Filter Symmetry:</label>
                <select id="symFilter" onchange="filterTable()">
                    <option value="ALL">All States</option>
                    <option value="Symmetrical">🟢 Symmetrical Only</option>
                    <option value="Asymmetrical">🔴 Asymmetrical Only</option>
                    <option value="Unassigned">⚪ Pending / Unassigned</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="zoneFilter">Zone:</label>
                <select id="zoneFilter" onchange="filterTable()">
                    <option value="ALL">All Zones</option>
                    <option value="Zone 1: Pass">Zone 1: Pass (≤ 1.5%) ({z1_tools})</option>
                    <option value="Zone 2: Manual">Zone 2: Manual (1.5% - 4.0%) ({z2_tools})</option>
                    <option value="Zone 3: Reject">Zone 3: Reject (> 4.0%) ({z3_tools})</option>
                    <option value="Excluded">Excluded ({excl_tools})</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="condFilter">Condition:</label>
                <select id="condFilter" onchange="filterTable()">
                    <option value="ALL">All Conditions</option>
                    <option value="new">New</option>
                    <option value="used">Used</option>
                    <option value="worn">Worn</option>
                    <option value="deposit">Deposit (BUE)</option>
                    <option value="fractured">Fractured</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="typeFilter">Type:</label>
                <select id="typeFilter" onchange="filterTable()">
                    <option value="ALL">All Types</option>
                    <option value="endmill">Endmill</option>
                    <option value="drill">Drill</option>
                    <option value="chamfer">Chamfer</option>
                    <option value="reamer">Reamer</option>
                    <option value="topper">Topper (Tap)</option>
                    <option value="t_slot">T-Slot</option>
                </select>
            </div>

            <button class="btn-reset" onclick="resetFilters()">Reset Filters</button>
            <div class="counter-badge" id="rowCount">Showing {total_tools} of {total_tools} tools</div>
        </div>

        <!-- Table with EXACT REQUESTED COLUMN ORDER -->
        <div class="table-wrapper">
            <table id="toolsTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0, 'str')">Tool ID <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(1, 'str')">Status <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(2, 'str')">Symmetrical <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(3, 'num')">Mean DSI (%) <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(4, 'num')">Max DSI (%) <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(5, 'str')">Zone <span class="sort-icon">▲▼</span></th>
                        <th>View</th>
                        <th onclick="sortTable(7, 'str')">Type <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(8, 'num')">Flute Number <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(9, 'str')">Org Condition <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(10, 'str')">Work Condition <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(11, 'str')">Physical Notes & Edits <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(12, 'num')">Dia (mm) <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(13, 'num')">Max Diff (px) <span class="sort-icon">▲▼</span></th>
                        <th onclick="sortTable(14, 'num')">ROI Height <span class="sort-icon">▲▼</span></th>
                    </tr>
                </thead>
                <tbody id="tableBody">
"""

        for _, r in full_df.iterrows():
            is_excl = not r["is_kept"]
            row_cls = "row-excluded" if is_excl else ""
            tid = r["tool_id"]

            cond_grp = r["working_condition"]
            c_badge = f"badge-{cond_grp}"
            s_badge = "badge-status-kept" if r["is_kept"] else "badge-status-excl"

            p_dsi_val = r["peak_dsi"]
            p_dsi_str = f"{p_dsi_val:.2f}%" if pd.notna(p_dsi_val) else "—"

            m_dsi_val = r["mean_dsi"]
            m_dsi_str = f"{m_dsi_val:.2f}%" if pd.notna(m_dsi_val) else "—"

            diff_val = r["max_pixel_diff"]
            diff_str = f"{diff_val:,.0f}" if pd.notna(diff_val) else "—"

            roi_val = r["roi_height_px"]
            roi_str = f"{roi_val:.0f}px" if pd.notna(roi_val) else "—"

            img_path = r[image_path_key]
            inspect_btn = f'<button class="btn-inspect" onclick="openPreviewModal(\'{img_path}\', \'{tid}\')">View</button>' if r["img_exists"] else "—"

            html += f"""                    <tr class="{row_cls}" data-tool-id="{tid}">
                        <td class="tool-id-cell">{tid}</td>
                        <td class="status-cell"><span class="badge {s_badge}">{r["status"]}</span></td>
                        <td class="sym-cell">
                            <select class="symmetry-select sym-unassigned" data-tool-id="{tid}" onchange="onSymmetryChange(this)">
                                <option value="" class="opt-unassigned" selected>⚪ — Select —</option>
                                <option value="Symmetrical" class="opt-sym" style="background-color: #064e3b; color: #4ade80; font-weight: bold;">🟢 Symmetrical</option>
                                <option value="Asymmetrical" class="opt-asym" style="background-color: #7f1d1d; color: #fca5a5; font-weight: bold;">🔴 Asymmetrical</option>
                            </select>
                        </td>
                        <td class="mean-dsi-cell dsi-cell">{m_dsi_str}</td>
                        <td class="max-dsi-cell dsi-cell">{p_dsi_str}</td>
                        <td class="zone-cell"><span class="badge {r['zone_badge']}">{r["zone_clean"]}</span></td>
                        <td class="view-cell">{inspect_btn}</td>
                        <td class="type-cell">{r["tool_type"]}</td>
                        <td class="flutes-cell">{r["edges"]}</td>
                        <td class="org-cond-cell"><span class="badge badge-{r['original_condition']}">{r["original_condition"]}</span></td>
                        <td class="work-cond-cell"><span class="badge {c_badge}">{r["working_condition"]}</span></td>
                        <td class="notes-cell">
                            <div class="editable-note" contenteditable="true" data-tool-id="{tid}" oninput="onNoteInput(this)" title="Click to edit physical notes">{r["notes"]}</div>
                        </td>
                        <td class="dia-cell">{r["diameter_mm"]:.1f}</td>
                        <td class="diff-cell">{diff_str}</td>
                        <td class="roi-cell">{roi_str}</td>
                    </tr>
"""

        html += """                </tbody>
            </table>
        </div>

        <div class="footer-bar">
            <div>Rotational Tool Condition Monitoring • Google Antigravity AAC Pipeline</div>
            <div>Selections auto-saved in browser cache. Press <strong>Ctrl+S</strong> or click <strong>Save HTML</strong> to persist to disk.</div>
        </div>
    </div>

    <!-- Image Preview Modal -->
    <div id="imageModal" class="modal-overlay" onclick="closeModalOnBackdrop(event)">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">Tool Inspection</h3>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <a id="modalDirectLink" href="#" target="_blank" class="btn-inspect">Open in New Tab</a>
                    <button class="modal-close" onclick="closeModal()">&times;</button>
                </div>
            </div>
            <div class="modal-body">
                <img id="modalImg" src="" alt="Tool Overlay Plot">
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast">Saved locally</div>

    <!-- Client-Side JavaScript for Interactivity, Persistence & Exports -->
    <script>
        const STORAGE_KEY = "tool_symmetry_user_annotations_v2";
        let currentSortCol = -1;
        let currentSortAsc = true;
        let noteDebounceTimer = null;

        // Toast feedback
        function showToast(msg) {
            const toast = document.getElementById("toast");
            if (!toast) return;
            toast.innerText = msg;
            toast.style.display = "block";
            clearTimeout(window._toastTimer);
            window._toastTimer = setTimeout(() => {
                toast.style.display = "none";
            }, 2400);
        }

        // Modal Preview
        function openPreviewModal(imgSrc, toolId) {
            document.getElementById("modalTitle").innerText = `${toolId} — Stacked Overlay & DSI Alignment`;
            document.getElementById("modalImg").src = imgSrc;
            document.getElementById("modalDirectLink").href = imgSrc;
            document.getElementById("imageModal").style.display = "flex";
        }

        function closeModal() {
            document.getElementById("imageModal").style.display = "none";
        }

        function closeModalOnBackdrop(e) {
            if (e.target.id === "imageModal") closeModal();
        }

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") closeModal();
        });

        // Local Storage Utilities
        function getStoredData() {
            try {
                const item = localStorage.getItem(STORAGE_KEY);
                return item ? JSON.parse(item) : {};
            } catch(e) {
                return {};
            }
        }

        function saveToolProperty(toolId, key, value) {
            const data = getStoredData();
            if (!data[toolId]) data[toolId] = {};
            data[toolId][key] = value;
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
            } catch(e) {
                console.error("Storage error:", e);
            }
        }

        // Apply visual styling to select and row based on symmetry choice
        function applySymmetryStyle(selectEl, row, val) {
            if (val === "Symmetrical") {
                selectEl.className = "symmetry-select sym-symmetrical";
                if (row) {
                    row.classList.add("row-sym");
                    row.classList.remove("row-asym");
                }
            } else if (val === "Asymmetrical") {
                selectEl.className = "symmetry-select sym-asymmetrical";
                if (row) {
                    row.classList.add("row-asym");
                    row.classList.remove("row-sym");
                }
            } else {
                selectEl.className = "symmetry-select sym-unassigned";
                if (row) {
                    row.classList.remove("row-sym", "row-asym");
                }
            }
        }

        // Symmetry Selection Change
        // NOTE: We intentionally DO NOT call filterTable() here!
        // The user's row must never vanish while they are working on it.
        function onSymmetryChange(selectEl) {
            const tid = selectEl.getAttribute("data-tool-id");
            const val = selectEl.value;
            const row = selectEl.closest("tr");

            applySymmetryStyle(selectEl, row, val);
            saveToolProperty(tid, "symmetry", val);
            updateKPICounters();
            showToast(`${tid} marked as ${val || 'Unassigned'}`);
        }

        // Editable Note Change
        function onNoteInput(divEl) {
            const tid = divEl.getAttribute("data-tool-id");
            const text = divEl.innerText.trim();
            clearTimeout(noteDebounceTimer);
            noteDebounceTimer = setTimeout(() => {
                saveToolProperty(tid, "notes", text);
                showToast(`Note saved for ${tid}`);
            }, 500);
        }

        // Restore User Annotations on Load
        function loadStoredAnnotations() {
            const data = getStoredData();
            console.log("Loading stored annotations for tools:", Object.keys(data).length);
            Object.keys(data).forEach(tid => {
                const row = document.querySelector(`tr[data-tool-id="${tid}"]`);
                if (!row) return;
                const rec = data[tid];
                if (rec.symmetry !== undefined) {
                    const sel = row.querySelector(".symmetry-select");
                    if (sel) {
                        sel.value = rec.symmetry;
                        applySymmetryStyle(sel, row, rec.symmetry);
                    }
                }
                if (rec.notes !== undefined) {
                    const noteDiv = row.querySelector(".editable-note");
                    if (noteDiv) {
                        noteDiv.innerText = rec.notes;
                    }
                }
            });
            updateKPICounters();
        }

        // Update Symmetrical / Asymmetrical / Pending KPI Badges
        function updateKPICounters() {
            let symCount = 0;
            let asymCount = 0;
            let pendingCount = 0;

            const rows = document.querySelectorAll("#tableBody tr[data-tool-id]");
            rows.forEach(row => {
                const sel = row.querySelector(".symmetry-select");
                if (sel) {
                    if (sel.value === "Symmetrical") symCount++;
                    else if (sel.value === "Asymmetrical") asymCount++;
                    else pendingCount++;
                }
            });

            const symEl = document.getElementById("kpiSymCount");
            const asymEl = document.getElementById("kpiAsymCount");
            const pendingEl = document.getElementById("kpiPendingCount");

            if (symEl) symEl.innerText = symCount;
            if (asymEl) asymEl.innerText = asymCount;
            if (pendingEl) pendingEl.innerText = pendingCount;
        }

        // Reset User Edits to Baseline
        function resetUserEdits() {
            if (confirm("Clear all your symmetry selections and custom notes back to empty baseline?")) {
                localStorage.removeItem(STORAGE_KEY);
                location.reload();
            }
        }

        // Save Current State Directly into Downloadable HTML (Ctrl+S support)
        function saveHtmlToDisk() {
            // Synchronize select and notes directly into DOM attributes before serialization
            const rows = document.querySelectorAll("#tableBody tr[data-tool-id]");
            rows.forEach(row => {
                const sel = row.querySelector(".symmetry-select");
                if (sel) {
                    const val = sel.value;
                    Array.from(sel.options).forEach(opt => {
                        if (opt.value === val) opt.setAttribute("selected", "selected");
                        else opt.removeAttribute("selected");
                    });
                    if (val === "Symmetrical") sel.setAttribute("class", "symmetry-select sym-symmetrical");
                    else if (val === "Asymmetrical") sel.setAttribute("class", "symmetry-select sym-asymmetrical");
                    else sel.setAttribute("class", "symmetry-select sym-unassigned");
                }
            });

            const fullHtml = "<!DOCTYPE html>\\n" + document.documentElement.outerHTML;
            const blob = new Blob([fullHtml], { type: "text/html;charset=utf-8" });
            const link = document.createElement("a");
            link.download = "tool_condition_summary_dashboard.html";
            link.href = URL.createObjectURL(blob);
            link.click();
            showToast("💾 Saved updated HTML file! (Overwrite in repo to git commit)");
        }

        // Keyboard Shortcut: Ctrl+S / Cmd+S triggers HTML save
        document.addEventListener("keydown", function(e) {
            if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
                e.preventDefault();
                saveHtmlToDisk();
            }
        });

        // Live Filtering (triggered by user typing in search or changing filter dropdowns)
        function filterTable() {
            const searchVal = (document.getElementById("searchInput") ? document.getElementById("searchInput").value : "").toLowerCase().trim();
            const statusVal = document.getElementById("statusFilter") ? document.getElementById("statusFilter").value : "ALL";
            const symVal = document.getElementById("symFilter") ? document.getElementById("symFilter").value : "ALL";
            const zoneVal = document.getElementById("zoneFilter") ? document.getElementById("zoneFilter").value : "ALL";
            const condVal = (document.getElementById("condFilter") ? document.getElementById("condFilter").value : "ALL").toLowerCase();
            const typeVal = (document.getElementById("typeFilter") ? document.getElementById("typeFilter").value : "ALL").toLowerCase();

            const rows = document.querySelectorAll("#tableBody tr[data-tool-id]");
            let visibleCount = 0;

            rows.forEach(row => {
                const text = row.innerText.toLowerCase();
                const statusEl = row.querySelector(".status-cell");
                const status = statusEl ? statusEl.innerText.trim() : "";

                const symEl = row.querySelector(".symmetry-select");
                const sym = symEl ? symEl.value : "";

                const zoneEl = row.querySelector(".zone-cell");
                const zone = zoneEl ? zoneEl.innerText.trim() : "";

                const typeEl = row.querySelector(".type-cell");
                const type = typeEl ? typeEl.innerText.trim().toLowerCase() : "";

                const condEl = row.querySelector(".work-cond-cell");
                const cond = condEl ? condEl.innerText.trim().toLowerCase() : "";

                const noteEl = row.querySelector(".editable-note");
                const note = noteEl ? noteEl.innerText.toLowerCase() : "";

                let matchesSearch = !searchVal || text.includes(searchVal) || sym.toLowerCase().includes(searchVal) || note.includes(searchVal);
                let matchesStatus = (statusVal === "ALL") || (status === statusVal);

                let matchesSym = true;
                if (symVal === "ALL") {
                    matchesSym = true;
                } else if (symVal === "Unassigned") {
                    matchesSym = (sym === "" || sym === "Unassigned");
                } else {
                    matchesSym = (sym === symVal);
                }

                let matchesZone = (zoneVal === "ALL") || (zone.includes(zoneVal));
                let matchesCond = (condVal === "ALL") || (cond === condVal);
                let matchesType = (typeVal === "ALL") || (type === typeVal);

                if (matchesSearch && matchesStatus && matchesSym && matchesZone && matchesCond && matchesType) {
                    row.style.display = "";
                    visibleCount++;
                } else {
                    row.style.display = "none";
                }
            });

            const countEl = document.getElementById("rowCount");
            if (countEl) countEl.innerText = `Showing ${visibleCount} of ${rows.length} tools`;

            // Friendly empty state row
            let emptyRow = document.getElementById("noResultsRow");
            if (visibleCount === 0) {
                if (!emptyRow) {
                    emptyRow = document.createElement("tr");
                    emptyRow.id = "noResultsRow";
                    emptyRow.innerHTML = `<td colspan="15" style="text-align: center; padding: 40px; color: var(--text-muted); font-size: 0.95rem;">
                        No tools match current filters. <button onclick="resetFilters()" class="btn-inspect" style="margin-left: 10px; cursor: pointer;">Reset Filters</button>
                    </td>`;
                    document.getElementById("tableBody").appendChild(emptyRow);
                }
                emptyRow.style.display = "";
            } else if (emptyRow) {
                emptyRow.style.display = "none";
            }
        }

        function resetFilters() {
            if (document.getElementById("searchInput")) document.getElementById("searchInput").value = "";
            if (document.getElementById("statusFilter")) document.getElementById("statusFilter").value = "ALL";
            if (document.getElementById("symFilter")) document.getElementById("symFilter").value = "ALL";
            if (document.getElementById("zoneFilter")) document.getElementById("zoneFilter").value = "ALL";
            if (document.getElementById("condFilter")) document.getElementById("condFilter").value = "ALL";
            if (document.getElementById("typeFilter")) document.getElementById("typeFilter").value = "ALL";
            filterTable();
        }

        // Sorting
        function sortTable(colIdx, type) {
            const table = document.getElementById("toolsTable");
            const tbody = document.getElementById("tableBody");
            const rows = Array.from(tbody.querySelectorAll("tr[data-tool-id]"));
            const ths = table.querySelectorAll("th");

            if (currentSortCol === colIdx) {
                currentSortAsc = !currentSortAsc;
            } else {
                currentSortCol = colIdx;
                currentSortAsc = true;
            }

            ths.forEach((th, idx) => {
                th.classList.remove("sorted-asc", "sorted-desc");
                if (idx === colIdx) {
                    th.classList.add(currentSortAsc ? "sorted-asc" : "sorted-desc");
                }
            });

            rows.sort((a, b) => {
                let cellA = "", cellB = "";
                if (colIdx === 2) {
                    const selA = a.querySelector(".symmetry-select");
                    const selB = b.querySelector(".symmetry-select");
                    cellA = selA ? selA.value : "";
                    cellB = selB ? selB.value : "";
                } else if (colIdx === 11) {
                    const nA = a.querySelector(".editable-note");
                    const nB = b.querySelector(".editable-note");
                    cellA = nA ? nA.innerText.trim() : "";
                    cellB = nB ? nB.innerText.trim() : "";
                } else {
                    cellA = a.children[colIdx] ? a.children[colIdx].innerText.trim() : "";
                    cellB = b.children[colIdx] ? b.children[colIdx].innerText.trim() : "";
                }

                if (type === "num") {
                    let numA = parseFloat(cellA.replace(/[^0-9.-]/g, "")) || 0;
                    let numB = parseFloat(cellB.replace(/[^0-9.-]/g, "")) || 0;
                    return currentSortAsc ? numA - numB : numB - numA;
                } else {
                    return currentSortAsc ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
                }
            });

            rows.forEach(r => tbody.appendChild(r));
        }

        // Export Symmetry List to JSON
        function exportSymmetryJSON() {
            const rows = Array.from(document.querySelectorAll("#tableBody tr[data-tool-id]"));
            const symmetrical = [];
            const asymmetrical = [];
            const unassigned = [];
            const details = {};

            rows.forEach(row => {
                const tid = row.getAttribute("data-tool-id");
                const symEl = row.querySelector(".symmetry-select");
                const sym = symEl ? symEl.value : "";
                const noteEl = row.querySelector(".editable-note");
                const notes = noteEl ? noteEl.innerText.trim() : "";
                const status = row.querySelector(".status-cell") ? row.querySelector(".status-cell").innerText.trim() : "";
                const meanDsi = row.querySelector(".mean-dsi-cell") ? row.querySelector(".mean-dsi-cell").innerText.trim() : "";
                const maxDsi = row.querySelector(".max-dsi-cell") ? row.querySelector(".max-dsi-cell").innerText.trim() : "";
                const zone = row.querySelector(".zone-cell") ? row.querySelector(".zone-cell").innerText.trim() : "";
                const type = row.querySelector(".type-cell") ? row.querySelector(".type-cell").innerText.trim() : "";
                const flutes = row.querySelector(".flutes-cell") ? row.querySelector(".flutes-cell").innerText.trim() : "";
                const orgCond = row.querySelector(".org-cond-cell") ? row.querySelector(".org-cond-cell").innerText.trim() : "";
                const workCond = row.querySelector(".work-cond-cell") ? row.querySelector(".work-cond-cell").innerText.trim() : "";

                if (sym === "Symmetrical") {
                    symmetrical.push(tid);
                } else if (sym === "Asymmetrical") {
                    asymmetrical.push(tid);
                } else {
                    unassigned.push(tid);
                }

                details[tid] = {
                    tool_id: tid,
                    symmetry: sym || "Unassigned",
                    notes: notes,
                    status: status,
                    mean_dsi: meanDsi,
                    max_dsi: maxDsi,
                    zone: zone,
                    type: type,
                    flutes: parseInt(flutes) || 0,
                    original_condition: orgCond,
                    working_condition: workCond
                };
            });

            const exportObj = {
                study: "Tool Condition Monitoring & Symmetry Classification",
                timestamp: new Date().toISOString(),
                summary: {
                    total_tools: rows.length,
                    symmetrical_count: symmetrical.length,
                    asymmetrical_count: asymmetrical.length,
                    unassigned_count: unassigned.length
                },
                symmetrical_tools: symmetrical,
                asymmetrical_tools: asymmetrical,
                unassigned_tools: unassigned,
                tool_annotations: details
            };

            const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: "application/json" });
            const link = document.createElement("a");
            link.download = "tool_symmetry_classification.json";
            link.href = URL.createObjectURL(blob);
            link.click();
            showToast(`Exported: ${symmetrical.length} Sym, ${asymmetrical.length} Asym, ${unassigned.length} Pending`);
        }

        // Export Full Table to CSV
        function exportTableToCSV(filename) {
            let csv = [];
            const headers = [
                "Tool ID", "Status", "Symmetrical", "Mean DSI (%)", "Max DSI (%)",
                "Zone", "Type", "Flute Number", "Org Condition", "Work Condition",
                "Physical Notes & Edits", "Dia (mm)", "Max Diff (px)", "ROI Height"
            ];
            csv.push(headers.map(h => `"${h}"`).join(","));

            const bodyRows = document.querySelectorAll("#tableBody tr[data-tool-id]");
            bodyRows.forEach(row => {
                if (row.style.display !== "none") {
                    const tid = row.getAttribute("data-tool-id");
                    const status = row.querySelector(".status-cell") ? row.querySelector(".status-cell").innerText.trim() : "";
                    const sym = row.querySelector(".symmetry-select") ? (row.querySelector(".symmetry-select").value || "Unassigned") : "Unassigned";
                    const meanDsi = row.querySelector(".mean-dsi-cell") ? row.querySelector(".mean-dsi-cell").innerText.trim() : "";
                    const maxDsi = row.querySelector(".max-dsi-cell") ? row.querySelector(".max-dsi-cell").innerText.trim() : "";
                    const zone = row.querySelector(".zone-cell") ? row.querySelector(".zone-cell").innerText.trim() : "";
                    const type = row.querySelector(".type-cell") ? row.querySelector(".type-cell").innerText.trim() : "";
                    const flutes = row.querySelector(".flutes-cell") ? row.querySelector(".flutes-cell").innerText.trim() : "";
                    const orgCond = row.querySelector(".org-cond-cell") ? row.querySelector(".org-cond-cell").innerText.trim() : "";
                    const workCond = row.querySelector(".work-cond-cell") ? row.querySelector(".work-cond-cell").innerText.trim() : "";
                    const notes = row.querySelector(".editable-note") ? row.querySelector(".editable-note").innerText.trim() : "";
                    const dia = row.querySelector(".dia-cell") ? row.querySelector(".dia-cell").innerText.trim() : "";
                    const diff = row.querySelector(".diff-cell") ? row.querySelector(".diff-cell").innerText.trim() : "";
                    const roi = row.querySelector(".roi-cell") ? row.querySelector(".roi-cell").innerText.trim() : "";

                    const rowVals = [
                        tid, status, sym, meanDsi, maxDsi, zone,
                        type, flutes, orgCond, workCond, notes,
                        dia, diff, roi
                    ];
                    csv.push(rowVals.map(v => `"${v.replace(/"/g, '""')}"`).join(","));
                }
            });

            const csvBlob = new Blob([csv.join("\\n")], { type: "text/csv" });
            const link = document.createElement("a");
            link.download = filename;
            link.href = URL.createObjectURL(csvBlob);
            link.click();
            showToast("Downloaded " + filename);
        }

        // Guaranteed Initialization on Document Ready or Already Interactive
        function initApp() {
            loadStoredAnnotations();
        }

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", initApp);
        } else {
            initApp();
        }
    </script>
</body>
</html>
"""
        return html

    # 6. Generate HTML variants
    # For Git repo: /home/alifalah/Projects/Tool_Condition_Monitoring/tool_condition_summary_dashboard.html
    html_repo = build_html("img_rel_repo")
    repo_html_path = os.path.join(repo_dir, "tool_condition_summary_dashboard.html")
    with open(repo_html_path, "w", encoding="utf-8") as fp:
        fp.write(html_repo)

    # For DATA/reports/ and artifact
    html_reports = build_html("img_rel_reports")
    reports_html_path = os.path.join(reports_dir, "tool_condition_summary_dashboard.html")
    with open(reports_html_path, "w", encoding="utf-8") as fp:
        fp.write(html_reports)

    shutil.copy2(reports_html_path, os.path.join(max_dsi_dir, "tool_condition_summary_dashboard.html"))
    shutil.copy2(reports_html_path, os.path.join(artifact_dir, "tool_condition_summary_dashboard.html"))

    print("\n=== Interactive Dashboard Generation Complete! ===")
    print(f"Git Repo Dashboard: {repo_html_path}")
    print(f"Reports Dashboard:  {reports_html_path}")
    print(f"Markdown Table:     {md_out}")
    print(f"CSV Database:       {csv_out}")

if __name__ == "__main__":
    generate_dashboard()
