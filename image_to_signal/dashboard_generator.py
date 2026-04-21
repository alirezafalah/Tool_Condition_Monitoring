import os
import json
import pandas as pd
import webbrowser

def generate_dashboard(csv_path, masks_dir, output_dir, real_dir=None, output_filename="dashboard.html", auto_open=True):
    """
    Generates an interactive HTML dashboard with Plotly and vanilla JS.
    Supports toggling between masks and real images, and a side-by-side compare mode.
    """
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
        
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

    raw_path = None
    processed_path = None
    if "processed" in csv_path:
        processed_path = csv_path
        raw_path = csv_path.replace("_processed_data.csv", "_raw_data.csv")
    elif "raw" in csv_path:
        raw_path = csv_path
        processed_path = csv_path.replace("_raw_data.csv", "_processed_data.csv")
    else:
        raw_path = csv_path

    raw_data = read_csv_data(raw_path)
    processed_data = read_csv_data(processed_path)
    
    if not raw_data and not processed_data:
        print(f"Error: Could not read CSV data from {csv_path}")
        return False

    # Determine phase shift
    shift_amount = 0
    if raw_path and os.path.exists(raw_path):
        try:
            df_raw = pd.read_csv(raw_path)
            area_col_raw = 'Smoothed ROI Area' if 'Smoothed ROI Area' in df_raw.columns else 'ROI Area (Pixels)'
            if area_col_raw in df_raw.columns:
                shift_amount = df_raw[area_col_raw].idxmin()
        except: pass

    # Process mask images
    mask_images = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg'))])
    mask_paths = ["file:///" + os.path.join(masks_dir, f).replace("\\", "/") for f in mask_images]
            
    # Process real images
    real_paths = []
    has_real = False
    if real_dir and os.path.isdir(real_dir):
        has_real = True
        real_images = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg'))])
        real_paths = ["file:///" + os.path.join(real_dir, f).replace("\\", "/") for f in real_images]
    else:
        real_paths = mask_paths # Fallback
            
    chart_data = {
        'raw': raw_data,
        'processed': processed_data,
        'has_raw': bool(raw_data),
        'has_processed': bool(processed_data),
        'shift_amount': shift_amount,
        'masks': mask_paths,
        'reals': real_paths,
        'has_real': has_real
    }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Condition Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
            width: 100%;
            max-width: 1600px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title {{
            flex: 1;
            text-align: left;
        }}
        .header-title h1 {{
            color: #4CAF50;
            margin-bottom: 5px;
            margin-top: 0;
        }}
        .header-title p {{
            margin: 0;
            color: #888;
        }}
        .controls {{
            display: flex;
            gap: 15px;
        }}
        button {{
            background: #2d2d2d;
            color: #fff;
            border: 1px solid #4CAF50;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }}
        button:hover {{
            background: #4CAF50;
        }}
        button.active {{
            background: #4CAF50;
            color: white;
        }}
        .container {{
            display: flex;
            width: 100%;
            max-width: 1600px;
            gap: 20px;
            background: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
            box-sizing: border-box;
            flex-direction: column;
        }}
        .top-row {{
            display: flex;
            width: 100%;
            gap: 20px;
        }}
        .chart-container {{
            flex: 2;
            position: relative;
            min-height: 450px;
        }}
        .images-wrapper {{
            flex: 1.2;
            display: flex;
            gap: 15px;
        }}
        .image-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #252525;
            border-radius: 8px;
            padding: 15px;
            min-height: 450px;
            border: 1px solid #333;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 350px;
            object-fit: contain;
            border: 2px solid #444;
            background: #000;
            border-radius: 4px;
        }}
        .slider-section {{
            width: 100%;
            background: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-sizing: border-box;
            border: 1px solid #333;
        }}
        input[type=range] {{
            width: 90%;
            margin-top: 10px;
            accent-color: #4CAF50;
        }}
        .slider-row {{
            width: 100%;
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .slider-label {{
            width: 100px;
            font-size: 1.1rem;
            font-weight: bold;
            color: #4CAF50;
            text-align: right;
            padding-right: 15px;
        }}
        .slider-input-wrap {{
            flex: 1;
            display: flex;
            align-items: center;
        }}
        .info-text {{
            color: #888;
            font-size: 0.95rem;
            margin-top: 10px;
        }}
        .image-info {{
            margin-top: 15px;
            font-size: 1.1rem;
            color: #ccc;
            font-weight: 500;
            text-align: center;
        }}
        
        /* Hide element class */
        .hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>🎭 Tool Condition Monitoring Dashboard</h1>
            <p>Interactive 1D Signal and Frame Visualization</p>
        </div>
        <div class="controls">
            <button id="btn-toggle-data" class="hidden">📊 View Raw Data</button>
            <button id="btn-toggle-img" class="{'' if has_real else 'hidden'}">👁️ View Real Image</button>
            <button id="btn-compare">🔄 Compare Mode: OFF</button>
        </div>
    </div>

    <div class="container">
        <div class="top-row">
            <div class="chart-container" id="plotly-chart"></div>
            
            <div class="images-wrapper" id="images-wrapper">
                <!-- Primary Image -->
                <div class="image-container" id="img-cont-1">
                    <h3 style="margin-top: 0; color: #ddd; margin-bottom: 10px;" id="title-1">Frame 1</h3>
                    <img id="image-1" src="" alt="Image 1">
                    <div id="info-1" class="image-info">Angle: -- | Area: --</div>
                </div>
                
                <!-- Secondary Image (Compare Mode) -->
                <div class="image-container hidden" id="img-cont-2">
                    <h3 style="margin-top: 0; color: #ddd; margin-bottom: 10px; color: #64B5F6;" id="title-2">Frame 2</h3>
                    <img id="image-2" src="" alt="Image 2">
                    <div id="info-2" class="image-info">Angle: -- | Area: --</div>
                </div>
            </div>
        </div>
        
        <div class="slider-section">
            <div class="slider-row" id="slider-row-1">
                <div class="slider-label" id="label-1">Angle: 0°</div>
                <div class="slider-input-wrap">
                    <input type="range" id="slider-1" min="0" max="100" value="0">
                </div>
            </div>
            
            <div class="slider-row hidden" id="slider-row-2">
                <div class="slider-label" style="color: #64B5F6;" id="label-2">Angle: 0°</div>
                <div class="slider-input-wrap">
                    <input type="range" id="slider-2" min="0" max="100" value="0" style="accent-color: #64B5F6;">
                </div>
            </div>
            <div class="info-text">Drag the sliders or click on the plot to update frames.</div>
        </div>
    </div>

    <script>
        const data = {json.dumps(chart_data)};
        
        let compareMode = false;
        let viewReal = false;
        let viewProcessed = false; // Default to raw
        
        let currentData = data.has_raw ? data.raw : data.processed; // fallback to processed if raw missing
        
        let idx1 = 0;
        let idx2 = Math.min(90, currentData.angles.length - 1); // Default offset
        
        const chartDiv = document.getElementById('plotly-chart');
        
        // DOM Elements
        const btnToggleData = document.getElementById('btn-toggle-data');
        const btnToggleImg = document.getElementById('btn-toggle-img');
        const btnCompare = document.getElementById('btn-compare');
        
        const imgCont2 = document.getElementById('img-cont-2');
        const sliderRow2 = document.getElementById('slider-row-2');
        
        const img1 = document.getElementById('image-1');
        const info1 = document.getElementById('info-1');
        const slider1 = document.getElementById('slider-1');
        const label1 = document.getElementById('label-1');
        
        const img2 = document.getElementById('image-2');
        const info2 = document.getElementById('info-2');
        const slider2 = document.getElementById('slider-2');
        const label2 = document.getElementById('label-2');
        
        // Setup data toggle button
        if (data.has_raw && data.has_processed) {{
            btnToggleData.classList.remove('hidden');
            btnToggleData.innerText = "📊 View Processed Data";
        }}

        function getImgSrc(i) {{
            const arr = viewReal ? data.reals : data.masks;
            if (arr.length === 0) return "";
            if (i < 0 || i >= currentData.angles.length) return "";
            
            const shift = viewProcessed ? data.shift_amount : 0;
            const actualIdx = (i + shift) % arr.length;
            return arr[actualIdx];
        }}

        function updateUI() {{
            // Update 1
            if (idx1 >= 0 && idx1 < currentData.angles.length) {{
                const a1 = currentData.angles[idx1];
                img1.src = getImgSrc(idx1);
                info1.innerText = "Angle: " + a1.toFixed(1) + "° | Area: " + Math.round(currentData.areas[idx1]).toLocaleString();
                label1.innerText = "Angle: " + a1.toFixed(1) + "°";
                slider1.value = idx1;
            }}
            
            // Update 2
            if (compareMode && idx2 >= 0 && idx2 < currentData.angles.length) {{
                const a2 = currentData.angles[idx2];
                img2.src = getImgSrc(idx2);
                info2.innerText = "Angle: " + a2.toFixed(1) + "° | Area: " + Math.round(currentData.areas[idx2]).toLocaleString();
                label2.innerText = "Angle: " + a2.toFixed(1) + "°";
                slider2.value = idx2;
            }}
            
            updatePlotlyLines();
        }}
        
        function updatePlotlyLines() {{
            const shapes = [];
            
            // Line 1
            if (idx1 >= 0 && idx1 < currentData.angles.length) {{
                shapes.push({{
                    type: 'line',
                    x0: currentData.angles[idx1],
                    y0: 0,
                    x1: currentData.angles[idx1],
                    y1: 1,
                    yref: 'paper',
                    line: {{ color: '#FF5252', width: 2, dash: 'dot' }}
                }});
            }}
            
            // Line 2
            if (compareMode && idx2 >= 0 && idx2 < currentData.angles.length) {{
                shapes.push({{
                    type: 'line',
                    x0: currentData.angles[idx2],
                    y0: 0,
                    x1: currentData.angles[idx2],
                    y1: 1,
                    yref: 'paper',
                    line: {{ color: '#64B5F6', width: 2, dash: 'dot' }}
                }});
            }}
            
            Plotly.relayout(chartDiv, {{ shapes: shapes }});
        }}

        // Initialize Plotly
        const trace = {{
            x: currentData.angles,
            y: currentData.areas,
            type: 'scatter',
            mode: 'lines+markers',
            marker: {{ size: 4, color: '#81C784' }},
            line: {{ color: '#4CAF50', width: 2 }},
            name: 'ROI Area',
            hovertemplate: 'Angle: %{{x}}°<br>Area: %{{y:.0f}}<extra></extra>'
        }};

        const layout = {{
            title: {{
                text: '1D Signal: ROI Area vs Angle',
                font: {{ color: '#e0e0e0', size: 18 }}
            }},
            paper_bgcolor: '#1e1e1e',
            plot_bgcolor: '#1e1e1e',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{ title: 'Angle (Degrees)', gridcolor: '#333', zerolinecolor: '#444' }},
            yaxis: {{ title: 'Area (Pixels)', gridcolor: '#333', zerolinecolor: '#444' }},
            hovermode: 'closest',
            margin: {{ l: 70, r: 30, t: 60, b: 60 }},
            clickmode: 'event+select'
        }};

        Plotly.newPlot(chartDiv, [trace], layout, {{responsive: true}});

        // Event Listeners
        slider1.addEventListener('input', (e) => {{
            idx1 = parseInt(e.target.value);
            updateUI();
        }});
        
        slider2.addEventListener('input', (e) => {{
            idx2 = parseInt(e.target.value);
            updateUI();
        }});
        
        // Plotly Click (to set points)
        chartDiv.on('plotly_click', (eventData) => {{
            if (eventData.points.length > 0) {{
                const i = eventData.points[0].pointIndex;
                if (compareMode && eventData.event.shiftKey) {{
                    idx2 = i;
                }} else {{
                    idx1 = i;
                }}
                updateUI();
            }}
        }});
        
        // Buttons
        btnToggleData.addEventListener('click', () => {{
            viewProcessed = !viewProcessed;
            currentData = viewProcessed ? data.processed : data.raw;
            
            if (viewProcessed) {{
                btnToggleData.innerText = "📊 View Raw Data";
                btnToggleData.classList.add('active');
            }} else {{
                btnToggleData.innerText = "📊 View Processed Data";
                btnToggleData.classList.remove('active');
            }}
            
            // Update chart data and force autorange
            Plotly.update(chartDiv, {{
                x: [currentData.angles],
                y: [currentData.areas]
            }}, {{
                'yaxis.autorange': true
            }});
            
            // Update slider max and indices if needed
            const maxIdx = currentData.angles.length - 1;
            slider1.max = maxIdx;
            slider2.max = maxIdx;
            if (idx1 > maxIdx) idx1 = 0;
            if (idx2 > maxIdx) idx2 = 0;
            
            updateUI();
        }});

        btnToggleImg.addEventListener('click', () => {{
            viewReal = !viewReal;
            if (viewReal) {{
                btnToggleImg.innerText = "👁️ View Binary Mask";
                btnToggleImg.classList.add('active');
            }} else {{
                btnToggleImg.innerText = "👁️ View Real Image";
                btnToggleImg.classList.remove('active');
            }}
            updateUI();
        }});
        
        btnCompare.addEventListener('click', () => {{
            compareMode = !compareMode;
            if (compareMode) {{
                btnCompare.innerText = "🔄 Compare Mode: ON";
                btnCompare.classList.add('active');
                imgCont2.classList.remove('hidden');
                sliderRow2.classList.remove('hidden');
            }} else {{
                btnCompare.innerText = "🔄 Compare Mode: OFF";
                btnCompare.classList.remove('active');
                imgCont2.classList.add('hidden');
                sliderRow2.classList.add('hidden');
            }}
            updateUI();
        }});
        
        // Initial load
        slider1.max = currentData.angles.length - 1;
        slider2.max = currentData.angles.length - 1;
        updateUI();
    </script>
</body>
</html>
"""

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, output_filename)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"✅ Dashboard generated at: {html_path}")
    
    if auto_open:
        try:
            webbrowser.open('file://' + os.path.abspath(html_path).replace('\\', '/'))
            return True
        except Exception as e:
            print(f"Error opening browser: {e}")
            return False
    return True
