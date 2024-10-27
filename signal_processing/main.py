import pandas as pd
from utils.smoothing import *

data = pd.read_csv(r"../data/intact.csv")
degree = data['Degree']
pixel_values = data['Sum of Pixels']

smoothed_pixels = smoothing_moving_average(degree, pixel_values, 5)

