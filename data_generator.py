import pandas as pd
import matplotlib.pyplot as plt

# Load the two tables
table_broken = pd.read_csv(r'data\table(broken).csv')  # Replace with your actual path
table_intact = pd.read_csv(r'data\table(intact).csv')  # Replace with your actual path

# Extract the values from table_broken between degrees [219, 309] (zero-indexed)
table_a_slice = table_broken[(table_broken['Degree'] >= 219) & (table_broken['Degree'] <= 309)].copy()

# Add 818 to the selected Sum of Pixels values
table_a_slice['Sum of Pixels'] += 250



# Adjust the indices of table_intact for degrees [51, 141] (0-indexed degrees [141, 231])
table_intact.loc[(table_intact['Degree'] >= 50) & (table_intact['Degree'] <= 140), 'Sum of Pixels'] = table_a_slice['Sum of Pixels'].values

# Save the modified table_intact if needed
table_intact.to_csv(r'data\modified_table_b.csv', index=False)

# Plot the modified table_intact
plt.figure(figsize=(8, 6))
plt.plot(table_intact['Degree'], table_intact['Sum of Pixels'], color='red', linewidth=1)
plt.title("Fractured Tool Pattern")
plt.xlabel("Degrees")
plt.ylabel("Area")
plt.ylim(1700, 2600)  # Set Y-axis limits
plt.grid(True)
plt.show()
