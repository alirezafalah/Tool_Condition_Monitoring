# Tool Condition Monitoring

This repository contains the code and data for analyzing CNC tool wear through image processing.

## Project Structure & Usage

* **`image_to_signal/`**: This is the primary, current pipeline and should be used for analysis.
* **`signal_processing/`** and **`tool_monitoring/`**: These are legacy versions of the project, kept for historical reference.
* **`tables/`**: This directory contains some CSV data from earlier experiments.

## How to Run the Main Pipeline

The main pipeline is designed to be run as a module from the project's root directory.

1.  Navigate to the `Tool_Condition_Monitoring` folder in your terminal.
2. Make sure you have your config in main set. 
3.  Execute the following command.

```bash
python -m image_to_signal.main --input_dir path/to/your/images
```

The input directory can contain either raw or blurred images. If only raw images are present, the script will automatically create the blurred versions before proceeding with the analysis.
