import step1_blur_and_rename
import step2_generate_masks
import step3_analyze_and_plot

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MASTER CONTROL PANEL
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Set these to True or False to run or skip steps
RUN_STEP_1_BLUR_AND_RENAME = True
RUN_STEP_2_GENERATE_MASKS = True
RUN_STEP_3_ANALYZE_AND_PLOT = True

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION PARAMETERS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

CONFIG = {
    # --- Directory Paths ---
    'RAW_DIR': 'data/tool069gain10paperBG',
    'BLURRED_DIR': 'data/tool069gain10paperBG_blurred',
    'FINAL_MASKS_DIR': 'data/tool069gain10paperBG_final_masks',
    'ROI_CSV_PATH': 'data/tool069gain10paperBG_area_vs_angle.csv',
    'ROI_PLOT_PATH': 'data/tool069gain10paperBG_area_vs_angle_plot.png',

    # --- Image Processing Parameters ---
    'blur_kernel': 13,
    'hue_min': 142,        # Corresponds to 200 degrees
    'hue_max': 163,        # Corresponds to 230 degrees
    'sat_min': 38,         # Corresponds to 15%
    'lab_ab_max': 126,       # Corresponds to a/b < -2 (currently unused)
    'closing_kernel': 5,

    # --- Data Analysis Parameters ---
    'images_for_366_deg': 372, # Because we used 5 Rev/min and we recorded for 12.2 seconds, therefore we have 366 degrees for each tool.
    'roi_height': 500,
    'outlier_std_dev_factor': 1.0,
}

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# EXECUTION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def main():
    """
    Orchestrates the entire image processing and analysis pipeline.
    """
    if RUN_STEP_1_BLUR_AND_RENAME:
        print("\n--- Running Step 1: Blur and Rename ---")
        step1_blur_and_rename.run(CONFIG)
        print("--- Step 1 Complete ---")

    if RUN_STEP_2_GENERATE_MASKS:
        print("\n--- Running Step 2: Generate Final Masks ---")
        step2_generate_masks.run(CONFIG)
        print("--- Step 2 Complete ---")

    if RUN_STEP_3_ANALYZE_AND_PLOT:
        print("\n--- Running Step 3: Analyze ROI and Plot ---")
        step3_analyze_and_plot.run(CONFIG)
        print("--- Step 3 Complete ---")
        
    print("\nPipeline finished.")

if __name__ == "__main__":
    main()
