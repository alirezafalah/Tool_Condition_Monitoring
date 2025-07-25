# Tool Wear Image Segmentation Pipeline

This project provides an automated pipeline to process a series of TIFF images of a rotating tool. It segments the tool from the background, calculates its projected area at each angle, and generates a final plot of area vs. rotation angle.

---

## Project Structure

The project is organized into a main pipeline and a `utils` directory containing helper functions.

```

image\_to\_signal/
│
├── **init**.py
├── main.py                     \# \<-- The main script to run the pipeline
├── step1\_blur\_and\_rename.py
├── step2\_generate\_masks.py
├── step3\_analyze\_and\_plot.py
│
├── data/                         \# \<-- Directory for input and output data
│
└── utils/
├── **init**.py
├── filters.py                \# \<-- Core image processing functions
├── image\_utils.py
└── mask\_refinement.py        \# \<-- Sandbox for finding new masks

````

---

## Execution

1.  **Place Data**: Put your raw `.tiff` image sequence into the directory specified by `RAW_DIR` in `main.py`.
2.  **Configure Pipeline**: Open `main.py` to:
    * Set which steps to run using the `MASTER CONTROL PANEL` booleans (`RUN_STEP_1`, etc.).
    * Adjust any parameters in the `CONFIG` dictionary as needed.
3.  **Run**: Execute the main script from the parent directory (`Tool_Condition_Monitoring` in your case).

    ```bash
    # From the 'Tool_Condition_Monitoring' directory
    python -m image_to_signal.main
    ```

---

## Workflow for Mask Refinement 🔬

Finding the perfect mask is an iterative process. This pipeline is designed to make that process efficient and organized.

### Step 1: Discover (The Playground)

* **File**: `utils/mask_refinement.py`
* **Purpose**: This is your interactive sandbox. It loads a single sample image and lets you experiment with different thresholds and logical combinations in the "CONTROL PANEL" and "Combine Masks" sections. It provides immediate visual feedback, allowing you to quickly see what works.
* **How to Run**:
    ```bash
    # From the 'Tool_Condition_Monitoring' directory
    python -m image_to_signal.utils.mask_refinement
    ```

### Step 2: Tune Values (The "Control Panel")

* **File**: `main.py`
* **Purpose**: Once the playground helps you find better **threshold values** (e.g., you decide the Saturation range should be 20-65% instead of 15-70%), you should update them here.
* **Action**: Modify the threshold values in the `CONFIG` dictionary. This is the main place for tuning parameters.

### Step 3: Change Logic (The "Engine Room")

* **File**: `utils/filters.py`
* **Purpose**: This is where the core algorithm lives. You should only edit this file if the playground helps you discover a fundamentally new **logical rule** for combining the masks.
* **Action**: Go to the `create_multichannel_mask` function and modify the single line that combines the boolean masks (e.g., changing `(b_mask) | (a_mask)` to `(b_mask) & (a_mask)`).
````

## Important Note
This project is structured as a Python package to keep the code organized and modular. Consequently, you must run all scripts from the project's root directory (Tool_Condition_Monitoring) using the python -m flag, which tells Python to run the code as a module. This approach is essential for the internal relative imports to function correctly. For example, execute python -m image_to_signal.main to run the entire pipeline, or python -m image_to_signal.utils.mask_refinement to use the experimental playground.