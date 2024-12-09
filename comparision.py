import numpy as np

# Suppose we have a baseline segment's coefficients:
baseline_coeffs = [0.531818 , 2.712956, -0.411131 , 0.367125]
baseline_coeffs = np.array([0.531818 , 2.712956, -0.411131 , 0.367125])

# Suppose we have a dictionary of coefficients for other segments:
segments = {
    'Segment2': [1.237892, 1.508812, 0.450256, -0.440028],
    'Segment3': [0.489337, 2.845055, -0.502670, 0.376174],
    'Segment4': [0.533613, 2.713403, -0.434978, 0.357810]
#     ...
}
segments = {
    'Segment2': np.array([1.237892, 1.508812, 0.450256, -0.440028]),
    'Segment3': np.array([0.489337, 2.845055, -0.502670, 0.376174]),
    'Segment4': np.array([0.533613, 2.713403, -0.434978, 0.357810])
}

# Calculate percentage differences
differences = {}
for seg_name, coeffs in segments.items():
    diff = ((coeffs - baseline_coeffs) / baseline_coeffs) * 100
    differences[seg_name] = diff

# 'differences' now holds the percentage differences for each segment’s coefficients.
print(differences)
