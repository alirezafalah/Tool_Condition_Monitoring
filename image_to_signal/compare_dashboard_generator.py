import os
import json
import pandas as pd
import webbrowser

def read_csv_data(p_csv):
    if not p_csv or not os.path.exists(p_csv): return None
    try:
        d = pd.read_csv(p_csv)
        if 'Angle (Degrees)' not in d.columns: return None
        ac = 'Smoothed ROI Area' if 'Smoothed ROI Area' in d.columns else 'ROI Area (Pixels)'
        if ac not in d.columns: return None
        return {
            'angles': d['Angle (Degrees)'].tolist(),
            'areas': d[ac].tolist()
        }
    except: return None

def extract_tool_data(masks_dir):
    """Extracts raw data, processed data, shift amount, and mask images for a tool."""
    tool_id = os.path.basename(os.path.normpath(masks_dir))
    analysis_dir = os.path.join(masks_dir, "analysis")
    
    raw_path = os.path.join(analysis_dir, f"{tool_id}_raw_data.csv")
    processed_path = os.path.join(analysis_dir, f"{tool_id}_processed_data.csv")
    
    # Fallback if specific tool id not found, just look for any raw/processed csv
    if not os.path.exists(raw_path):
        csvs = [f for f in os.listdir(analysis_dir) if f.endswith('_raw_data.csv')] if os.path.exists(analysis_dir) else []
        if csvs: raw_path = os.path.join(analysis_dir, csvs[0])
    if not os.path.exists(processed_path):
        csvs = [f for f in os.listdir(analysis_dir) if f.endswith('_processed_data.csv')] if os.path.exists(analysis_dir) else []
        if csvs: processed_path = os.path.join(analysis_dir, csvs[0])

    raw_data = read_csv_data(raw_path)
    processed_data = read_csv_data(processed_path)
    
    shift_amount = 0
    if raw_path and os.path.exists(raw_path):
        try:
            df_raw = pd.read_csv(raw_path)
            ac = 'Smoothed ROI Area' if 'Smoothed ROI Area' in df_raw.columns else 'ROI Area (Pixels)'
            if ac in df_raw.columns:
                shift_amount = df_raw[ac].idxmin()
        except: pass

    mask_images = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg'))])
    mask_paths = ["file:///" + os.path.join(masks_dir, f).replace("\\", "/") for f in mask_images]
    
    return {
        'id': tool_id,
        'raw': raw_data,
        'processed': processed_data,
        'has_raw': bool(raw_data),
        'has_processed': bool(processed_data),
        'shift_amount': int(shift_amount),
        'masks': mask_paths
    }

def generate_comparison_dashboard(dir_A, dir_B, output_dir, output_filename="compare_dashboard.html", auto_open=True):
    if not os.path.isdir(dir_A) or not os.path.isdir(dir_B):
        print("Error: Invalid tool directories provided.")
        return False
        
    tool_A = extract_tool_data(dir_A)
    tool_B = extract_tool_data(dir_B)
    
    if not tool_A['has_raw'] and not tool_A['has_processed']:
        print(f"Error: Could not find CSV data for Tool A in {dir_A}/analysis")
        return False
    if not tool_B['has_raw'] and not tool_B['has_processed']:
        print(f"Error: Could not find CSV data for Tool B in {dir_B}/analysis")
        return False
        
    chart_data = {
        'tool_A': tool_A,
        'tool_B': tool_B,
        'has_processed': tool_A['has_processed'] and tool_B['has_processed'],
        'has_raw': tool_A['has_raw'] and tool_B['has_raw']
    }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .header h1 {{ margin: 0; font-size: 1.8rem; color: #fff; }}
        .header p {{ margin: 5px 0 0 0; color: #aaa; }}
        .controls button {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        .controls button:hover {{ background: rgba(255, 255, 255, 0.15); }}
        .controls button.active {{ background: #4CAF50; border-color: #4CAF50; }}
        .hidden {{ display: none !important; }}
        
        .container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .chart-panel {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .images-panel {{
            display: flex;
            gap: 20px;
            justify-content: center;
        }}
        .image-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .image-card h3 {{
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.2rem;
            text-align: center;
            word-break: break-all;
        }}
        .image-card img {{
            max-width: 100%;
            max-height: 350px;
            object-fit: contain;
            border-radius: 8px;
            background: #000;
        }}
        .image-info {{
            margin-top: 15px;
            font-size: 1.1rem;
            color: #ccc;
            font-weight: 500;
        }}
        .slider-section {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .slider-section label {{
            font-size: 1.1rem;
            font-weight: bold;
            color: #fff;
            min-width: 100px;
        }}
        input[type="range"] {{
            flex: 1;
            height: 6px;
            background: #333;
            border-radius: 3px;
            appearance: none;
            outline: none;
            accent-color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚖️ Multi-Tool Comparison</h1>
            <p>Side-by-side analysis of edge geometries</p>
        </div>
        <div class="controls">
            <button id="btn-toggle-data" class="hidden">📊 View Raw Data</button>
        </div>
    </div>

    <div class="container">
        <!-- Plotly Chart -->
        <div class="chart-panel">
            <div id="plotly-chart" style="width: 100%; height: 500px;"></div>
        </div>
        
        <!-- Master Slider -->
        <div class="slider-section">
            <label id="slider-label">Angle: 0°</label>
            <input type="range" id="master-slider" min="0" max="100" value="0">
        </div>
        
        <!-- Image Viewers -->
        <div class="images-panel">
            <!-- Tool A -->
            <div class="image-card">
                <h3 style="color: #64B5F6;">Tool A: {tool_A['id']}</h3>
                <img id="img-A" src="" alt="Tool A">
                <div id="info-A" class="image-info">Angle: -- | Area: --</div>
            </div>
            
            <!-- Tool B -->
            <div class="image-card">
                <h3 style="color: #FFB74D;">Tool B: {tool_B['id']}</h3>
                <img id="img-B" src="" alt="Tool B">
                <div id="info-B" class="image-info">Angle: -- | Area: --</div>
            </div>
        </div>
    </div>

    <script>
        const data = {json.dumps(chart_data)};
        
        let viewProcessed = false; // Always default to Raw
        let currentIdx = 0;
        
        const chartDiv = document.getElementById('plotly-chart');
        const slider = document.getElementById('master-slider');
        const sliderLabel = document.getElementById('slider-label');
        const btnToggleData = document.getElementById('btn-toggle-data');
        
        const imgA = document.getElementById('img-A');
        const infoA = document.getElementById('info-A');
        const imgB = document.getElementById('img-B');
        const infoB = document.getElementById('info-B');
        
        if (data.has_raw && data.has_processed) {{
            btnToggleData.classList.remove('hidden');
            btnToggleData.innerText = "📊 View Processed Data";
        }}
        
        function getActiveData(tool) {{
            return viewProcessed ? tool.processed : tool.raw;
        }}
        
        function getImgSrc(tool, idx) {{
            if (tool.masks.length === 0) return "";
            const shift = viewProcessed ? tool.shift_amount : 0;
            const actualIdx = (idx + shift) % tool.masks.length;
            return tool.masks[actualIdx];
        }}
        
        function updateUI() {{
            const dataA = getActiveData(data.tool_A);
            const dataB = getActiveData(data.tool_B);
            
            // Assume both have roughly same angles, but use max available
            const maxAngles = Math.max(
                dataA ? dataA.angles.length : 0, 
                dataB ? dataB.angles.length : 0
            );
            
            if (currentIdx >= maxAngles) currentIdx = 0;
            
            // Determine the display angle (using A as primary if exists)
            let currentAngle = 0;
            if (dataA && currentIdx < dataA.angles.length) currentAngle = dataA.angles[currentIdx];
            else if (dataB && currentIdx < dataB.angles.length) currentAngle = dataB.angles[currentIdx];
            
            sliderLabel.innerText = "Angle: " + currentAngle.toFixed(1) + "°";
            slider.value = currentIdx;
            
            // Tool A
            if (dataA && currentIdx < dataA.angles.length) {{
                imgA.src = getImgSrc(data.tool_A, currentIdx);
                infoA.innerText = "Area: " + Math.round(dataA.areas[currentIdx]).toLocaleString();
            }} else {{
                imgA.src = "";
                infoA.innerText = "No Data";
            }}
            
            // Tool B
            if (dataB && currentIdx < dataB.angles.length) {{
                imgB.src = getImgSrc(data.tool_B, currentIdx);
                infoB.innerText = "Area: " + Math.round(dataB.areas[currentIdx]).toLocaleString();
            }} else {{
                imgB.src = "";
                infoB.innerText = "No Data";
            }}
            
            // Update vertical line
            Plotly.relayout(chartDiv, {{
                shapes: [{{
                    type: 'line',
                    x0: currentAngle, y0: 0,
                    x1: currentAngle, y1: 1,
                    yref: 'paper',
                    line: {{ color: '#fff', width: 2, dash: 'dot' }}
                }}]
            }});
        }}
        
        function buildPlot() {{
            const dataA = getActiveData(data.tool_A);
            const dataB = getActiveData(data.tool_B);
            
            const traces = [];
            
            if (dataA) {{
                traces.push({{
                    x: dataA.angles,
                    y: dataA.areas,
                    type: 'scatter', mode: 'lines+markers',
                    name: 'Tool A',
                    line: {{ color: '#64B5F6', width: 2 }},
                    marker: {{ size: 4, color: '#64B5F6' }},
                    hovertemplate: 'Tool A<br>Angle: %{{x}}°<br>Area: %{{y:.0f}}<extra></extra>'
                }});
            }}
            
            if (dataB) {{
                traces.push({{
                    x: dataB.angles,
                    y: dataB.areas,
                    type: 'scatter', mode: 'lines+markers',
                    name: 'Tool B',
                    line: {{ color: '#FFB74D', width: 2 }},
                    marker: {{ size: 4, color: '#FFB74D' }},
                    hovertemplate: 'Tool B<br>Angle: %{{x}}°<br>Area: %{{y:.0f}}<extra></extra>'
                }});
            }}
            
            const layout = {{
                title: {{ text: 'Signal Comparison: Tool A vs Tool B', font: {{ color: '#e0e0e0', size: 18 }} }},
                paper_bgcolor: '#1e1e1e', plot_bgcolor: '#1e1e1e',
                font: {{ color: '#e0e0e0' }},
                xaxis: {{ title: 'Angle (Degrees)', gridcolor: '#333', zerolinecolor: '#444' }},
                yaxis: {{ title: 'Area (Pixels)', gridcolor: '#333', zerolinecolor: '#444' }},
                hovermode: 'closest',
                margin: {{ l: 70, r: 30, t: 60, b: 40 }},
                clickmode: 'event+select',
                legend: {{ orientation: 'h', y: -0.15 }}
            }};
            
            Plotly.newPlot(chartDiv, traces, layout, {{responsive: true}});
            
            // Slider bounds
            const maxA = dataA ? dataA.angles.length - 1 : 0;
            const maxB = dataB ? dataB.angles.length - 1 : 0;
            slider.max = Math.max(maxA, maxB);
        }}
        
        btnToggleData.addEventListener('click', () => {{
            viewProcessed = !viewProcessed;
            if (viewProcessed) {{
                btnToggleData.innerText = "📊 View Raw Data";
                btnToggleData.classList.add('active');
            }} else {{
                btnToggleData.innerText = "📊 View Processed Data";
                btnToggleData.classList.remove('active');
            }}
            buildPlot();
            updateUI();
        }});
        
        slider.addEventListener('input', (e) => {{
            currentIdx = parseInt(e.target.value);
            updateUI();
        }});
        
        chartDiv.on('plotly_click', (eventData) => {{
            if (eventData.points.length > 0) {{
                currentIdx = eventData.points[0].pointIndex;
                updateUI();
            }}
        }});
        
        // Init
        buildPlot();
        updateUI();
    </script>
</body>
</html>
"""

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, output_filename)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Comparison Dashboard generated at: {html_path}")
    
    if auto_open:
        try:
            webbrowser.open('file://' + os.path.abspath(html_path).replace('\\', '/'))
            return True
        except Exception as e:
            print(f"Error opening browser: {e}")
            return False
    return True
