"""
Interactive Matplotlib widget for displaying tool profile with degree indicator.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import numpy as np


class MatplotlibPlotWidget(QWidget):
    def __init__(self, csv_path, is_processed=False):
        super().__init__()
        
        self.csv_path = csv_path
        self.is_processed = is_processed
        self.current_degree = None
        self.show_degree_line = False
        
        # Create matplotlib figure with responsive size
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Load and plot data
        self.load_and_plot()
    
    def load_and_plot(self):
        """Load CSV data and create the plot."""
        if not os.path.exists(self.csv_path):
            return
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        
        # Clear and create plot
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Determine column to plot
        if self.is_processed:
            y_column = 'ROI Area (Pixels)'
            ylabel = 'Normalized ROI Area (0-1)'
            title = 'Processed Tool Profile'
        else:
            # Check if smoothed data exists
            if 'Smoothed ROI Area' in self.df.columns:
                y_column = 'Smoothed ROI Area'
                ylabel = 'Projected Area in ROI (Pixel Count)'
            else:
                y_column = 'ROI Area (Pixels)'
                ylabel = 'Projected Area in ROI (Pixel Count)'
            title = 'Tool ROI Area vs. Rotation Angle'
        
        # Plot the data
        self.ax.plot(self.df['Angle (Degrees)'], self.df[y_column], 
                    marker='.', linestyle='-', markersize=4, color='blue', label='Data')
        
        # Styling
        self.ax.set_title(title, fontsize=18, fontweight='bold')
        self.ax.set_xlabel('Angle (Degrees)', fontsize=14)
        self.ax.set_ylabel(ylabel, fontsize=14)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, 360)
        self.ax.set_xticks(np.arange(0, 361, 30))
        
        # Store the vertical line object
        self.vline = None
        
        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()
    
    def set_degree_indicator(self, show, degree=None):
        """Show/hide and update the vertical degree indicator line."""
        self.show_degree_line = show
        self.current_degree = degree
        
        # Remove existing line
        if self.vline:
            self.vline.remove()
            self.vline = None
        
        # Draw new line if enabled
        if show and degree is not None:
            self.vline = self.ax.axvline(x=degree, color='red', linewidth=2, 
                                        linestyle='-', label='Current Position', zorder=10)
        
        self.canvas.draw()
    
    def reload_csv(self, csv_path, is_processed=False):
        """Reload with a different CSV file."""
        self.csv_path = csv_path
        self.is_processed = is_processed
        self.load_and_plot()
        
        # Restore degree indicator if it was shown
        if self.show_degree_line and self.current_degree is not None:
            self.set_degree_indicator(True, self.current_degree)
