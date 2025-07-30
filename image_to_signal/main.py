from . import step1_blur_and_rename
from . import step2_generate_masks
from . import step3_analyze_and_plot

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MASTER CONTROL PANEL
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Set these to True or False to run or skip steps
RUN_STEP_1_BLUR_AND_RENAME = True
RUN_STEP_2_GENERATE_MASKS = False
RUN_STEP_3_ANALYZE_AND_PLOT = False

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION PARAMETERS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

CONFIG = {
    # --- Directory Paths ---
    'RAW_DIR': 'image_to_signal/data/tool014gain10paperBG',
    'BLURRED_DIR': 'image_to_signal/data/tool054gain10paperBG_blurred',
    'FINAL_MASKS_DIR': 'image_to_signal/data/tool054gain10paperBG_final_masks',
    'ROI_CSV_PATH': 'image_to_signal/data/tool014gain54paperBG_area_vs_angle.csv',
    'ROI_PLOT_PATH': 'image_to_signal/data/tool014gain54paperBG_area_vs_angle_plot.png',

    # --- Image Processing Parameters ---
    'blur_kernel': 13,
    'closing_kernel': 21,

    # -- HSV Parameters --
    'h_threshold_min': 70 // 2,        # Corresponds to 70 degrees (Currently not used)
    'h_threshold_max': 100 // 2,        # Corresponds to 100 degrees (Currently not used)
    's_threshold_min': 15 * 2.55,    # Saturation > 15% (Currently not used)
    's_threshold_max': 70 * 2.55,    # Saturation <= 70% (Currently not used)
    'V_threshold_min': 45 * 2.55,    # Value > 45%
    'V_threshold_max': 55 * 2.55,    # Value <= 55%

    # -- LAB Parameters --
    'L_threshold_min': 50 * 2.55,     # Direct from LAB space, corresponds to L = 0 (Currently not used)
    'L_threshold_max': 56 * 2.55,     # Direct from LAB space, corresponds to L = 50 (Currently not used)
    'a_threshold_min': -10 + 128,     # Corresponds to a* = -10 (min value) (Currently not used)
    'a_threshold_max': -1 + 128,     # Corresponds to a* = 127 (max value)
    'b_threshold_min': -10 + 128,      # Corresponds to b* = -10 (min value) (Currently not used)
    'b_threshold_max': -8 + 128,     # Corresponds to b* = -8 (max value)

    # --- Background Subtraction Parameters ---
    'BACKGROUND_SUBTRACTION_METHOD': 'lab', # Options: 'none', 'absdiff', 'lab'
    'APPLY_MULTICHANNEL_MASK': False,
    'BACKGROUND_IMAGE_PATH': 'image_to_signal/data/paper_background.tiff',
    'DIFFERENCE_THRESHOLD': 15,


    # --- Data Analysis Parameters ---
    'images_for_366_deg': 363, # Because we used 5 Rev/min and we recorded for 12.2 seconds, therefore we have 366 degrees for each tool.
    'roi_height': 300,
    'outlier_std_dev_factor': 3.0,
    'APPLY_MOVING_AVERAGE': True,  
    'MOVING_AVERAGE_WINDOW': 5,  
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
