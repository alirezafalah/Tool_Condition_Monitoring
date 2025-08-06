import pandas as pd
import numpy as np

# --- Configuration ---
# IMPORTANT: Replace this with the actual path to your ORIGINAL INTACT CSV file.
intact_file_path = '../chamfer_1edge_fractured.csv'

# Define the output path for the new, modified fractured data CSV.
output_fractured_path = 'chamfer_2edge_broken.csv'

# Define the parameters for the modifications.
general_variation = 20  # The +/- random variation for most points.
fracture_degree_start = 53 # The start of the new fracture zone.
fracture_degree_end = 125   # The end of the new fracture zone.
fracture_pixel_min = 2100  # The minimum pixel value in the new fracture zone.
fracture_pixel_max = 2200  # The maximum pixel value in the new fracture zone.

# --- Data Loading and Processing ---
try:
    # Load the original data from the intact tool's CSV file.
    df_intact = pd.read_csv(intact_file_path)
    print(f"Successfully loaded '{intact_file_path}'.")
    
    # Create a copy of the intact dataframe to modify.
    # This ensures the original data is not changed.
    df_fractured = df_intact.copy()

    # --- Apply Modifications ---
    # We will iterate through each row of the DataFrame to apply our logic.
    
    # Create a new list to hold the modified pixel values.
    new_pixel_values = []

    for index, row in df_fractured.iterrows():
        degree = row['Degree']
        original_pixels = row['Sum of Pixels']
        
        # Check if the current degree falls within the new fracture zone.
        if fracture_degree_start <= degree <= fracture_degree_end:
            # If it is in the zone, generate a completely new random value.
            new_value = np.random.randint(fracture_pixel_min, fracture_pixel_max + 1)
        else:
            # If it's outside the zone, apply the small, general variation.
            random_offset = np.random.randint(-general_variation, general_variation + 1)
            new_value = original_pixels + random_offset
            
        new_pixel_values.append(new_value)

    # Replace the 'Sum of Pixels' column with our newly generated values.
    df_fractured['Sum of Pixels'] = new_pixel_values
    
    # --- Save the New Data ---
    # Save the modified DataFrame to a new CSV file.
    # `index=False` prevents pandas from writing the row index as a column.
    df_fractured.to_csv(output_fractured_path, index=False)
    
    print(f"\nModifications complete!")
    print(f"New fractured data has been saved to '{output_fractured_path}'.")

except FileNotFoundError:
    print(f"Error: The file was not found at '{intact_file_path}'.")
    print("Please update the 'intact_file_path' variable with the correct location of your file.")
except KeyError as e:
    print(f"Error: A required column was not found in the CSV: {e}.")
    print("Please ensure your CSV has 'Degree' and 'Sum of Pixels' columns.")

