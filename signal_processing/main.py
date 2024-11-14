import pandas as pd
from utils.smoothing import *

data = "../data/drill_broken.csv"
# degree = data['Degree']
# pixel_values = data['Sum of Pixels']

smoothed_pixels = savgol_peak_finder(data)
# smoothed_pixels = smoothing_moving_average(degree, pixel_values, 5)
