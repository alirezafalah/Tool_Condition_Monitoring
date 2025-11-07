import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class InteractivePlotter:
    """
    An interactive plot editor for cleaning time-series data.

    Load a CSV, click and drag points vertically to correct their values,
    and close the window to save the changes to a new CSV file.
    """
    def __init__(self, input_csv, output_csv):
        self.input_csv_path = input_csv
        self.output_csv_path = output_csv
        self.selected_point = None
        self.load_data()

        # --- Plotting Setup ---
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.setup_plot()

        # --- Connect Matplotlib Events ---
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def load_data(self):
        """Loads data from the input CSV file."""
        try:
            self.df = pd.read_csv(self.input_csv_path)
            # Ensure column names are consistent
            self.x_col = self.df.columns[0]
            self.y_col = self.df.columns[1]
            print(f"Data loaded successfully from '{self.input_csv_path}'")
        except FileNotFoundError:
            print(f"Error: CSV file not found at '{self.input_csv_path}'.")
            self.df = None

    def setup_plot(self):
        """Initializes the plot elements."""
        if self.df is None: return
        
        # Main data line
        self.line, = self.ax.plot(self.df[self.x_col], self.df[self.y_col], 'o-', color='steelblue', markersize=5, picker=5)
        
        # Visual settings
        self.ax.set_title('Interactive Data Editor\n(Drag points vertically, close window to save)', fontsize=16)
        self.ax.set_xlabel(self.x_col, fontsize=12)
        self.ax.set_ylabel(self.y_col, fontsize=12)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        self.ax.grid(True)

    def update_plot(self):
        """Redraws the plot with updated data."""
        self.line.set_ydata(self.df[self.y_col])
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        """Handles mouse button press events to select a data point."""
        if event.inaxes != self.ax: return
        
        # Find the index of the closest point to the click
        distances = np.sqrt((self.df[self.x_col] - event.xdata)**2)
        closest_index = np.argmin(distances)
        
        # Check if the click was close enough to a point
        if distances[closest_index] < 1.0: # Tolerance in x-axis units (degrees)
            self.selected_point = closest_index
            print(f"Selected point index: {self.selected_point}")

    def on_motion(self, event):
        """Handles mouse movement to drag the selected point."""
        if self.selected_point is None or event.inaxes != self.ax: return
        
        # Update the y-value of the selected point in the DataFrame
        self.df.loc[self.selected_point, self.y_col] = event.ydata
        self.update_plot()

    def on_release(self, event):
        """Handles mouse button release to deselect the point."""
        if self.selected_point is not None:
            print(f"Released point index: {self.selected_point}")
            self.selected_point = None

    def on_close(self, event):
        """Saves the modified data when the plot window is closed."""
        if self.df is not None:
            self.df.to_csv(self.output_csv_path, index=False)
            print(f"\nPlot window closed. Corrected data saved to '{self.output_csv_path}'")

    def show(self):
        """Displays the plot window."""
        if self.df is not None:
            plt.show()

def main():
    """Main function to run the interactive plotter."""
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # CONTROL PANEL
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 1. Path to your original CSV file
    INPUT_CSV = 'image_to_signal/data/temp/fixed_interactive_data.csv'

    # 2. Path to save the new, corrected CSV file
    OUTPUT_CSV = 'image_to_signal/data/temp/fixed_interactive_data.csv'
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plotter = InteractivePlotter(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV)
    plotter.show()

if __name__ == "__main__":
    main()
