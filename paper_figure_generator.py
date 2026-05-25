import os
import sys
import glob
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def generate_paper_figure(tool_id, tool_type, state, edges):
    print(f"Generating paper figure for {tool_id} ({edges}-edge {tool_type}, {state})...")
    
    # Paths
    base_data = os.path.abspath("../DATA")
    tools_dir = os.path.join(base_data, "tools", tool_id)
    masks_dir = os.path.join(base_data, "masks", f"{tool_id}_final_masks")
    sym_dir = os.path.join(base_data, "symmetry", tool_id)
    half_dir = os.path.join(masks_dir, "half_tool_analysis")
    
    output_dir = os.path.join(base_data, "paper_outputs", tool_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Random Raw Image
    raw_images = glob.glob(os.path.join(tools_dir, "*.tiff")) + glob.glob(os.path.join(tools_dir, "*.png"))
    if not raw_images:
        print(f"Error: No raw images found in {tools_dir}")
        return
    
    selected_raw = random.choice(raw_images)
    raw_out = os.path.join(output_dir, f"{tool_id}_raw_sample.png")
    # Convert and resize for paper (don't need 16MB tiff)
    img = Image.open(selected_raw)
    img.thumbnail((800, 800))
    img.save(raw_out, "PNG")
    print(f"Saved raw sample to {raw_out}")

    # 2. Master Mask
    master_src = os.path.join(sym_dir, f"{tool_id}_master_mask.png")
    master_out = os.path.join(output_dir, f"{tool_id}_master_mask.png")
    if os.path.exists(master_src):
        import shutil
        shutil.copy2(master_src, master_out)
        print(f"Copied master mask to {master_out}")
    else:
        print(f"Warning: Master mask not found at {master_src}")

    # 3. ROI Visualization (CROP TOP 120 PIXELS)
    roi_src = os.path.join(sym_dir, f"{tool_id}_roi_visualization.png")
    roi_out = os.path.join(output_dir, f"{tool_id}_roi_visualization.png")
    if os.path.exists(roi_src):
        img_roi = cv2.imread(roi_src)
        if img_roi is not None:
            # Crop top 120 pixels to remove rotation artifacts AND old embedded titles
            cropped_roi = img_roi[120:, :]
            cv2.imwrite(roi_out, cropped_roi)
            print(f"Cropped and saved ROI visualization to {roi_out}")
    else:
        print(f"Warning: ROI visualization not found at {roi_src}")

    # 4. Refined Signal Graph (SCALED & SHIFTED, PDF with Big Fonts)
    csv_path = os.path.join(half_dir, "right_half_analysis.csv")
    graph_out = os.path.join(output_dir, f"{tool_id}_signal_graph.pdf")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # --- Pre-processing: Scale to 0-1 ---
        min_area = df['Area'].min()
        max_area = df['Area'].max()
        df['Area'] = (df['Area'] - min_area) / (max_area - min_area)
        
        # --- Pre-processing: Shift Minimum to 0 degrees ---
        min_idx = df['Area'].idxmin()
        df_shifted = pd.concat([df.loc[min_idx:], df.loc[:min_idx - 1]]).reset_index(drop=True)
        
        # Correct the angle column
        first_angle = df_shifted.iloc[0]['Angle']
        df_shifted['Angle'] = df_shifted['Angle'] - first_angle
        df_shifted.loc[df_shifted['Angle'] < 0, 'Angle'] += 360
        
        plt.figure(figsize=(8, 6))
        plt.plot(df_shifted['Angle'], df_shifted['Area'], color='blue', linewidth=2.5)
        
        # Big fonts for paper
        plt.title(f"{edges}-Edge {tool_type} Profile ({state})", fontsize=20, fontweight='bold')
        plt.xlabel("Angle (Degrees)", fontsize=18)
        plt.ylabel("Normalized Area (0-1)", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 360)
        
        plt.tight_layout()
        plt.savefig(graph_out, bbox_inches='tight')
        plt.close()
        print(f"Generated pre-processed high-quality graph to {graph_out}")
    else:
        print(f"Warning: CSV for graph not found at {csv_path}")

    # 5. LaTeX Snippet
    tex_path = os.path.join(output_dir, f"{tool_id}_figure.tex")
    
    caption_state = "functional" if state.lower() == "functional" else "fractured"
    main_caption = f"Visual analysis of a {caption_state} {edges}-edge {tool_type.lower()}."
    
    tex_content = r"""
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/""" + f"{tool_id}_raw_sample.png" + r"""}
        \caption{Raw image sample}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/""" + f"{tool_id}_master_mask.png" + r"""}
        \caption{Master mask}
    \end{subfigure}
    \vskip\baselineskip
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/""" + f"{tool_id}_roi_visualization.png" + r"""}
        \caption{ROI visualization}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/""" + f"{tool_id}_signal_graph.pdf" + r"""}
        \caption{Normalized signal profile}
    \end{subfigure}
    \caption{""" + main_caption + r"""}
    \label{fig:tool_""" + tool_id + r"""_analysis}
\end{figure}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Generated LaTeX snippet to {tex_path}")
    print("\n" + "="*50)
    print(f"DONE! All files for {tool_id} are in {output_dir}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures for a tool.")
    parser.add_argument("--tool", required=True, help="Tool ID (e.g., tool028)")
    parser.add_argument("--type", required=True, help="Tool Type (e.g., Drill)")
    parser.add_argument("--state", required=True, choices=["functional", "fractured"], help="State of the tool")
    parser.add_argument("--edges", type=int, default=2, help="Number of edges (e.g., 2)")
    
    args = parser.parse_args()
    generate_paper_figure(args.tool, args.type, args.state, args.edges)
