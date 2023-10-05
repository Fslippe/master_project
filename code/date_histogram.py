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
from scipy import ndimage
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
from tensorflow import keras    
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN
from pyhdf.SD import SD, SDC
import matplotlib as mpl
#tf.config.threading.set_inter_op_parallelism_threads(1)
from extract_training_data import *
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from pyhdf.error import HDF4Error
from functions import *
from autoencoder import SobelFilterLayer, SimpleAutoencoder
from tensorflow.keras.models import load_model


import joblib

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

bands=[29]
patch_size = 64
print(len(bands))
autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(29)_filter_encoder")
max_vals = np.array([15.703261])



folder = "/scratch/fslippe/modis/MOD02/daytime_1km/ /scratch/fslippe/modis/MOD02/boundary_1km/"
 
# start = "20201201"
# end = "20240430"
start = "20210401"
end = "20210430"
dates_converted = []

start_converted = convert_to_day_of_year(start)
end_converted = convert_to_day_of_year(end)
print(start_converted)
print(end_converted)
x, dates = extract_1km_data(folder, bands=bands, start_date=start_converted, end_date=end_converted)
#x, dates = zip(*[(xi, date) for xi, date in zip(x, dates) if (xi.shape[0] > 64) and (xi.shape[1] > 64)])
#x, dates = zip(*[(xi+2, date) for xi, date in zip(x, dates) if (xi.shape[0] > 64) and (xi.shape[1] > 64)])

x = list(x)
dates = list(dates)


cluster_map = []
all_patches = []
starts = []
ends =[]
shapes = []
start = 0 
print("extract patch")
for image in x:
    shapes.append(image.shape[0:2])
    patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
    all_patches.append(patches)
    starts.append(start)
    ends.append(start + len(patches))
    start += len(patches)

# Stack filtered patches from all images
patches = np.concatenate(all_patches, axis=0) / max_vals
encoded_patches = encoder.predict(patches)
encoded_patches_flat = encoded_patches.reshape(encoded_patches.shape[0], -1)

cluster = joblib.load('/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(29)_filter_cluster_daytime_lab1.pkl')
print("loaded cluster")
cluster_predict = cluster.predict(encoded_patches_flat)
labels = cluster_predict#.labels_


desired_label = 1
size_threshold = 7  # Adjust based on the minimum size of the region you are interested in
selected_dates = []
selected_images = []
global_min = np.min([np.min(cm) for cm in cluster.labels_])
global_max = np.max([np.max(cm) for cm in cluster.labels_])
norm = Normalize(vmin=global_min, vmax=global_max)  

print("calculating labels")
for i in range(len(x)):
    height, width = shapes[i]

    # Calculate the dimensions of the reduced resolution array
    reduced_height = height // patch_size
    reduced_width = width //patch_size
    
    current_labels = labels[starts[i]:ends[i]]
   
    label_map = (np.reshape(labels[starts[i]:ends[i]], (reduced_height, reduced_width)))

    binary_map = (label_map == desired_label)

    # Label connected components
    labeled_map, num_features = ndimage.label(binary_map)

    # Measure sizes of connected components
    region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

    # Check if any region exceeds the size threshold
    if any(region_sizes > size_threshold):

        if dates[i] not in selected_dates:
            selected_dates.append(dates[i]) 
            fig, axs = plt.subplots(1,2)
            fig.suptitle("CAO found for threshold %s" %(size_threshold))
            axs[0].imshow(x[i], cmap="gray")
            cb =axs[1].imshow(label_map, cmap="tab10", norm=norm)
            plt.colorbar(cb)
            plt.show()
            # plt.savefig("/uio/hume/student-u37/fslippe/master_project/figures/CAO_found_at_%s" %(dates[i])) 


np.save("/uio/hume/student-u37/fslippe/data/date_hist_2020_21_%s" %(size_threshold), np.array(selected_dates))
#np.save("/uio/hume/student-u37/fslippe/data/date_hist_202104+2_%s" %(size_threshold), np.array(selected_dates))