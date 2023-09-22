# %%
from keras.layers import Input, Dense, Flatten, Reshape
from sklearn.feature_extraction import image as sk_image
from concurrent.futures import ProcessPoolExecutor
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
#tf.config.threading.set_inter_op_parallelism_threads(1)
from extract_training_data import *
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from pyhdf.error import HDF4Error
from functions import *


# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#bands = [6, 7, 20, 28, 28, 31]


# %%
import importlib
import extract_training_data
importlib.reload(extract_training_data)
from extract_training_data import *
folder = "/scratch/fslippe/modis/MOD02/cao_test_data/"

start = "20210321"
end = "20230521"
dates = ["20210321"]
dates_converted = []
for date in dates:
    dates_converted.append(convert_to_day_of_year(date))

start_converted = convert_to_day_of_year(start)
end_converted = convert_to_day_of_year(end)
print(start_converted)
print(end_converted)
x = [xi for xi in  extract_1km_data(folder, bands=[1], start_date=start_converted, end_date=end_converted) if xi.shape[0] > 64]



#x = extract_250m_data(folder, bands=[1], start_date=start_converted, end_date=end_converted)
len(x)



# %%
for i in range(len(x)):
    print(np.mean(x[i]))
    print(x[i].shape)

# %%
import autoencoder
import importlib
importlib.reload(autoencoder)
from autoencoder import SobelFilterLayer, SimpleAutoencoder
bands = [1]
patch_size = 64


#normalized_patches = np.concatenate([autoencoder.extract_patches(n_d) for n_d in normalized_data], axis=0)


# %%
from autoencoder import SobelFilterLayer, SimpleAutoencoder
bands = [1]
patch_size = 64
from tensorflow.keras.models import load_model
print(len(bands))
autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)

encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_encoder")

# %%
# normalized_patches = np.concatenate([autoencoder.extract_patches(n_d) for n_d in normalized_data[:4]], axis=0)

gc.collect()
os.environ['OPENBLAS_NUM_THREADS'] = '1'


start = 3
index_list = [i for i in range(20)]#, 13, 14]
cluster_map_encoded = autoencoder_predict.kmeans([x[i] for i in index_list], n_clusters=10, encoder=encoder)
#cluster_map_encoded = autoencoder_predict.kmeans(x, n_clusters=10, encoder=encoder)

gc.collect()


# %%
cluster_map = []
all_patches = []
starts = []
ends =[]
shapes = []
start = 0 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

for image in [x[i] for i in index_list]:
    shapes.append(image.shape[0:2])
    patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
    all_patches.append(patches)
    starts.append(start)
    ends.append(start + len(patches))
    start += len(patches)



# Stack filtered patches from all images
patches = np.concatenate(all_patches, axis=0)

patches_flat = patches.reshape(patches.shape[0], -1)

# KMeans clustering
kmeans = KMeans(10).fit(patches_flat)

labels = kmeans.labels_

# Assuming your original data shape is (height, width)
for i in range(len(index_list)):#(len(x)):
    height, width = shapes[i]
    
    # Calculate the dimensions of the reduced resolution array
    reduced_height = height // patch_size
    reduced_width = width // patch_size
    cluster_map.append(np.reshape(labels[starts[i]:ends[i]], (reduced_height, reduced_width)))

gc.collect()


# %%
from matplotlib.colors import Normalize

# Determine global min and max labels
global_min = np.min([np.min(cm) for cm in cluster_map])
global_max = np.max([np.max(cm) for cm in cluster_map])

norm = Normalize(vmin=global_min, vmax=global_max)

for i in range(len(cluster_map)):
    fig, axs = plt.subplots(1, 3, figsize=[15, 8])
    print(np.mean(x[i]))
    cb = axs[0].imshow(cluster_map_encoded[i], cmap="tab10", norm=norm)
    plt.colorbar(cb, ax=axs[0])
    cb = axs[1].imshow(cluster_map[i], cmap="tab10", norm=norm)
    plt.colorbar(cb, ax=axs[1])
    
    cb = axs[2].imshow(x[i][::4, ::4, 0])
    plt.colorbar(cb, ax=axs[2])
    plt.tight_layout()

plt.show()

# %%



