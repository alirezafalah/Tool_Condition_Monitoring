import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def main():
    report_dir = '/home/alifalah/Projects/DATA/report/inference_time/'
    
    cpu_step2 = load_json(os.path.join(report_dir, 'step2_inference.json')).get('total_time_seconds', 139.9)
    cpu_step3 = load_json(os.path.join(report_dir, 'step3_master_mask_inference.json')).get('total_time_seconds', 6.4)
    cpu_step4 = load_json(os.path.join(report_dir, 'step4_offset_inference.json')).get('total_time_seconds', 39.2)
    
    scenarios = ["Local 12-Core CPU"]
    steps = ["Mask Generation", "Master Mask & Tilt Correction", "Offset Analysis & DSI Calculation"]
    
    mask_gen_times = [cpu_step2]
    master_mask_times = [cpu_step3]
    offset_times = [cpu_step4]
    
    data = [mask_gen_times, master_mask_times, offset_times]
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
            
    fig, ax = plt.subplots(figsize=(6, 8))
    
    colors = ['#4C72B0', '#55A868', '#C44E52']
    x = np.arange(len(scenarios))
    width = 0.4
    bottom = np.zeros(len(scenarios))
    
    for i, step_data in enumerate(data):
        bars = ax.bar(x, step_data, width, label=steps[i], bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.5)
        
        for j, bar in enumerate(bars):
            val = step_data[j]
            y_center = bar.get_y() + bar.get_height() / 2
            
            if val > 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    y_center, 
                    f"{val:.1f}s", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=12
                )
            else:
                y_offset_pts = -20 if i == 1 else 20
                ax.annotate(
                    f"{val:.1f}s",
                    xy=(bar.get_x() + bar.get_width(), y_center),
                    xytext=(30, y_offset_pts),
                    textcoords='offset points',
                    ha='left', va='center', color='black', fontweight='bold', fontsize=12,
                    arrowprops=dict(arrowstyle="-", color='black', lw=1)
                )
                
        bottom += step_data
        
    for j in range(len(scenarios)):
        ax.text(
            x[j], bottom[j] + 2, 
            f"Total: {bottom[j]:.1f}s", 
            ha='center', va='bottom', color='black', fontweight='bold', fontsize=14
        )
        
    ax.set_ylabel('Cumulative Inference Time (Seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Pipeline Inference Time\n(Local 12-Core CPU)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=14, fontweight='bold')
    
    # Legend placement outside or upper right
    ax.legend(fontsize=11, frameon=True, edgecolor='black', facecolor='white', loc='upper right')
    
    # Expand x-axis slightly so right-side labels don't get cut off
    ax.set_xlim(-0.5, 1.2)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_path = os.path.join(report_dir, 'inference_times_cpu_only.pdf')
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved to {out_path}")

if __name__ == '__main__':
    main()
