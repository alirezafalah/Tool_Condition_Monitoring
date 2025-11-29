import os
from . import step1_blur_and_rename
from . import step2_generate_masks
from . import step3_analyze_and_plot
from . import step4_process_and_plot

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MASTER CONTROL PANEL
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Set these to True or False to run or skip steps
RUN_STEP_1_BLUR_AND_RENAME = False
RUN_STEP_2_GENERATE_MASKS = False
RUN_STEP_3_ANALYZE_AND_PLOT = False  # Generate raw CSV/plot
RUN_STEP_4_PROCESS_AND_PLOT = True  # Generate processed CSV/plot


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION PARAMETERS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Calculate the path to the DATA folder (two levels up from this file, then into DATA)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Tool_Condition_Monitoring
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATA"))  # Go up to CCD_DATA, then into DATA

# --- Tool Identification ---
TOOL_ID = 'tool002'  # Set the tool ID here - all paths will be automatically derived

CONFIG = {
    # --- Directory Paths (Auto-generated from TOOL_ID) ---
    'RAW_DIR': os.path.join(DATA_ROOT, 'tools', TOOL_ID),
    'BLURRED_DIR': os.path.join(DATA_ROOT, 'blurred', f'{TOOL_ID}_blurred'),
    'FINAL_MASKS_DIR': os.path.join(DATA_ROOT, 'masks', f'{TOOL_ID}_final_masks'),
    'ROI_CSV_PATH': os.path.join(DATA_ROOT, '1d_profiles', f'{TOOL_ID}_raw_data.csv'),
    'ROI_PLOT_PATH': os.path.join(DATA_ROOT, '1d_profiles', f'{TOOL_ID}_raw_plot.svg'),
    'PROCESSED_CSV_PATH': os.path.join(DATA_ROOT, '1d_profiles', f'{TOOL_ID}_processed_data.csv'),
    'PROCESSED_PLOT_PATH': os.path.join(DATA_ROOT, '1d_profiles', f'{TOOL_ID}_processed_plot.svg'),
    'BACKGROUND_IMAGE_PATH': os.path.join(DATA_ROOT, 'backgrounds', 'paper_background.tiff'),

    # --- Processing Parameters (for Step 4) ---
    'NUMBER_OF_PEAKS': 2,  # Number of cutting edges/flutes to segment the data into

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
    'DIFFERENCE_THRESHOLD': 33,


    # --- Data Analysis Parameters ---
    'roi_height': 200,
    'WHITE_RATIO_OUTLIER_THRESHOLD': 0.8,
    'APPLY_MOVING_AVERAGE': True,  
    'MOVING_AVERAGE_WINDOW': 5,
    'NUMBER_OF_PEAKS': 3,

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
        print("\n--- Running Step 3: Analyze ROI and Plot (Raw Data) ---")
        step3_analyze_and_plot.run(CONFIG)
        print("--- Step 3 Complete ---")
    
    if RUN_STEP_4_PROCESS_AND_PLOT:
        print("\n--- Running Step 4: Process and Plot (Normalized & Segmented) ---")
        step4_process_and_plot.run(CONFIG)
        print("--- Step 4 Complete ---")
        
    print("\nPipeline finished.")

if __name__ == "__main__":
    main()
