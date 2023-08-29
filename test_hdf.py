import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import matplotlib as mpl
file_name = "../Downloads/MOD06_L2.A2022096.0725.061.2022096195112.hdf"

# Open the file.
hdf = SD(file_name, SDC.READ)

# List available datasets.
datasets = hdf.datasets()

for idx, sds in enumerate(datasets.keys()):
    print(idx, sds)

# Replace 'YOUR_DATASET_NAME' with the dataset you're interested in.
lon = hdf.select('Longitude')[:]
lat = hdf.select('Latitude')[:]
print(lon.shape)
data = hdf.select('cloud_top_temperature_1km')[:]
print(data.shape)
data = (np.where(data == -9999, np.nan, data))
plt.contourf(data)
plt.colorbar()
plt.show()
