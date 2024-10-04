import pandas as pd

# Load the broken tool data from a CSV file
file_path = '../data/table(broken).csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Define the range for the pattern (degrees 129 to 219)
pattern_start = 129
pattern_end = 219

# Extract the pattern between degrees 129 and 219
pattern = data[(data['Degree'] >= pattern_start) & (data['Degree'] <= pattern_end)].copy()

# Repeat the pattern over the entire degree range (360 degrees)
# Calculate how many full cycles of the pattern can fit
full_cycles = data.shape[0] // pattern.shape[0]
repeated_pattern = pd.concat([pattern] * full_cycles, ignore_index=True)

# Adjust the length of the repeated pattern to match the original dataset length
if len(repeated_pattern) < len(data):
    repeated_pattern = pd.concat([repeated_pattern, pattern[:len(data) - len(repeated_pattern)]], ignore_index=True)

repeated_pattern['Degree'] = data['Degree']

# Save the new table as a CSV file
output_file = 'new_tool_pattern_from_broken.csv'  
repeated_pattern.to_csv(output_file, index=False)

print(f"New tool data saved to {output_file}")
