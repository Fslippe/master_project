import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import matplotlib as mpl
from scipy.ndimage import zoom

# Read HDF
# Read HDF
hdf = SD("/scratch/fslippe/modis/MOD02/cao_test_data/MOD021KM.A2023063.0835.061.2023063191054.hdf", SDC.READ)
lat = hdf.select("Latitude")[:]
lon = hdf.select("Longitude")[:]

var = hdf.select("EV_1KM_RefSB")[:][0]

# Create a mask from lat where True indicates valid data (lat >= 60)
mask_lowres = (lat >= 60) #& (lat <= 82) & (lon > -35) & (lon < 35)


# Determine the zoom factor based on the shape of var and lat
zoom_factor_y = var.shape[0] / lat.shape[0]
zoom_factor_x = var.shape[1] / lat.shape[1]
zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions

# Upscale the mask to match var's resolution
mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # order=0 for nearest neighbor interpolation

# If you want to continue with the rows and columns that have at least one valid value:
valid_rows = np.any(mask_highres, axis=1)
valid_cols = np.any(mask_highres, axis=0)

var_trimmed = var[valid_rows][:, valid_cols]

plt.imshow(var_trimmed)
plt.show()
