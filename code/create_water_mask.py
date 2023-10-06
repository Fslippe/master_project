from pyhdf.SD import SD, SDC
import os
import matplotlib.pyplot as plt
import xarray as xr

folder = "/uio/hume/student-u37/fslippe/data/land_sea_ice_mask/"

print(xr.open_dataset(folder + "NSIDC-0780_SeaIceRegions_EASE2-N3.125km_v1.0.nc"))