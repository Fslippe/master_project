from keras.layers import Input, Dense, Flatten, Reshape
from sklearn.feature_extraction import image as sk_image
from keras.models import Model
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import convolve2d
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pyhdf.SD import SD, SDC
import matplotlib as mpl
#tf.config.threading.set_inter_op_parallelism_threads(128)

from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

from autoencoder import SobelFilterLayer, SimpleAutoencoder
folder = "/nird/projects/NS9600K/data/modis/cao/"
#folder = "/home/filip/Documents/master_project/data/MOD02/"
folder = "/uio/hume/student-u37/fslippe/data/cao/"
file_name = folder + "MOD021KM.A2021080.1300.061.2021081011315.hdf"

print(file_name)
hdf = SD(file_name, SDC.READ)
bands = [6, 7, 20, 30, 5]

list1 = [int(num_str) for num_str in hdf.select("EV_250_Aggr1km_RefSB").attributes()["band_names"].split(",")]
list2 = [int(num_str) for num_str in hdf.select("EV_500_Aggr1km_RefSB").attributes()["band_names"].split(",")]
list3 = [int(num_str) for num_str in hdf.select("EV_1KM_RefSB").attributes()["band_names"].split(",") if num_str.isdigit()]
list4 = [int(num_str) for num_str in hdf.select("EV_1KM_Emissive").attributes()["band_names"].split(",")]

file_layers = np.empty(36, dtype=object)
for i, (band) in enumerate(list1):
    file_layers[band-1] = {"EV_250_Aggr1km_RefSB": i}
for i, (band) in enumerate(list2):
    file_layers[band-1] = {"EV_500_Aggr1km_RefSB": i}    
for i, (band) in enumerate(list3):
    file_layers[band-1] = {"EV_1KM_RefSB": i}
for i, (band) in enumerate(list4):
    file_layers[band-1] = {"EV_1KM_Emissive": i}


#all_files = os.listdir(folder)[16:18]
all_files = os.listdir(folder)[4:5]

X = np.empty((len(all_files), 2030, 1354, len(bands)))

x = np.empty((2030, 1354, len(bands)))


for i, (file) in enumerate(all_files):
    hdf = SD(folder + file, SDC.READ)
    for j, (band) in enumerate(bands):
        key = list(file_layers[band-1].keys())[0]
        idx = list(file_layers[band-1].values())[0]

        attrs = hdf.select(key).attributes()
        data = hdf.select(key)[:][idx]
        is_nan = (np.where(data == attrs["_FillValue"]))
        data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]

        if not len(is_nan[0]) == 0:
            data = data[is_nan[0][-1]+1:, :] if is_nan[1][-1] == 1353 else data[:, is_nan[1][-1]+1:]

        X[i, :, :, j] = data[:2030, :1354]


plt.imshow(X[0,:,:,0], cmap="gray")

autoencoder = SimpleAutoencoder(3, 128, 128)
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
autoencoder.fit(X, epochs=200, batch_size=32, optimizer="adam", threshold=0.09,loss="combined")
# print(X[0].shape)
# #autoencoder = simple_autoencoder([data_01], patch_size)
# autoencoder = simple_autoencoder(1, (2040, 1354), patch_size)    
# autoencoder.fit(X, epochs=5, batch_size=256)