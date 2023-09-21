import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import matplotlib as mpl
file_name = "/uio/hume/student-u37/fslippe/data/nird_mount/MOD02QKM_202012-202104/MOD02QKM.A2021120.2340.061.2021121105137.hdf"


# Open the file.
hdf = SD(file_name, SDC.READ)

# List available datasets.
datasets = hdf.datasets()

for idx, sds in enumerate(datasets.keys()):
    print(idx, sds)

attrs = (hdf.select("EV_250_RefSB").attributes())

data = np.where(hdf.select("EV_250_RefSB")[:][0]==attrs["_FillValue"], np.nan, hdf.select("EV_250_RefSB")[:][0]/32767)
print(np.nanmax(data), np.nanmean(data))

# Replace 'YOUR_DATASET_NAME' with the dataset you're interested in.
