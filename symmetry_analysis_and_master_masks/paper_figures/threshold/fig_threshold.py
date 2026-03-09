import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
# Use the CSV you just created/cleaned
CSV_PATH = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA\threshold_analysis\left_right_method\left_right_analysis_results.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASENAME = "threshold_weibull_safety"

# We choose 10% PFA to prioritize Safety (catching all fractures)
TARGET_PFA = 0.10 

def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Filter for Functional Tools Only (New + Used)
    functional_tools = df[df['Condition'].str.lower().isin(['new', 'used'])]
    ratios = functional_tools['Mean Ratio'].values
    
    # Fit Weibull
    # shapes: (shape, loc, scale)
    shape, loc, scale = stats.weibull_min.fit(ratios, floc=0)
    print(f"Weibull Parameters: Shape={shape:.4f}, Scale={scale:.4f}")

    # Calculate Threshold
    # T = lambda * (-ln(PFA))^(1/k)
    threshold = scale * (-np.log(TARGET_PFA))**(1/shape)
    
    print(f"\n--- SAFETY THRESHOLD CALCULATION (PFA={TARGET_PFA}) ---")
    print(f"Calculated Threshold: {threshold:.6f}")
    
    # Check against data
    fractured = df[df['Condition'].str.lower().isin(['fractured', 'deposit'])]
    min_frac = fractured['Mean Ratio'].min()
    print(f"Smallest Fracture: {min_frac:.6f}")
    
    if threshold < min_frac:
        print("SUCCESS: Threshold is lower than all fractures (100% Sensitivity).")
    else:
        print(f"WARNING: Threshold misses fractures below {threshold:.6f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Histogram of functional tools
    plt.hist(ratios, bins=8, density=True, alpha=0.6, color='blue', label='Functional Tools (Hist)')
    
    # Weibull PDF
    x = np.linspace(0, max(ratios)*2.5, 100)
    pdf = stats.weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, 'r-', lw=2, label=f'Weibull Fit (k={shape:.2f})')
    
    # Threshold Line
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Safety Threshold (PFA={TARGET_PFA})')
    
    # Annotate the threshold value
    plt.text(threshold + 0.002, plt.ylim()[1]*0.8, f'T = {threshold:.3f}', color='green', fontweight='bold')

    plt.title(f'Probabilistic Thresholding (Safety Bias, PFA={TARGET_PFA})')
    plt.xlabel('Asymmetry Ratio')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ["png", "svg"]:
        out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {OUTPUT_BASENAME}.png/.svg")
    plt.close()

if __name__ == "__main__":
    main()