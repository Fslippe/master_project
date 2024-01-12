import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
from keras.layers import Input, Dense, Flatten, Reshape
from sklearn.feature_extraction import image as sk_image
from concurrent.futures import ProcessPoolExecutor
import cartopy.feature as cfeature
from keras.models import Model
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
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans, MiniBatchKMeans
import joblib
import plot_functions
import importlib 
importlib.reload(plot_functions)
from plot_functions import *
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
# Visualize the result
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,"  , len(logical_gpus), "Logical GPUs")
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import socket
hostname = socket.gethostname()
if "nird" in hostname:
    tf.config.threading.set_inter_op_parallelism_threads(8)
    data_loc = "/nird/projects/NS9600K/fslippe/data/"
    folder = "/nird/projects/NS9600K/data/modis/cao/MOD02/2020/ /nird/projects/NS9600K/data/modis/cao/MOD02/2021/ /nird/projects/NS9600K/data/modis/cao/MOD02/2023/"
if "mimi" in hostname:
    data_loc = "/uio/hume/student-u37/fslippe/data/"
    folder = "/scratch/fslippe/modis/MOD02/daytime_1km/ /scratch/fslippe/modis/MOD02/boundary_1km/ /scratch/fslippe/modis/MOD02/night_1km/ /scratch/fslippe/modis/MOD02/may-nov_2021/ /scratch/fslippe/modis/MOD02/cao_test_data/"


bands=[29]
patch_size = 128

dates_block = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_block.npy")
times_block = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_block.npy")
dates_rest = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_rest.npy")
times_rest = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_rest.npy")
dates = np.append(dates_block, dates_rest)
times = np.append(times_block, times_rest)

x_cao = []
masks_cao = []
lon_lats_cao = []

dates, times = dates_block[10:12], times_block[10:12]
s=0
for (d, m) in zip(dates, times):
    s+=1
    arr = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A%s.%s.combined.npy" %(d, m))
    x_cao.append(arr)
    arr = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A%s.%s.combined.npy" %(d, m))
    masks_cao.append(arr)
    arr = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A%s.%s.combined.npy" %(d, m))
    lon_lats_cao.append(arr)
    #print("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A%s.%s_combined" %(d, m))
    #idx = np.where((np.array(dates_cao) == d) & (np.array(mod_min_cao) == m))[0][0]
    #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A%s.%s.combined.npy" %(d, m), masks_cao[idx])
    #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A%s.%s.combined.npy" %(d, m), lon_lats_cao[idx])
    #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/new_files/MOD021KM.A%s.%s.combined" %(d, m), arr)
    

max_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/max_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
min_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/min_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
strides = 32    #patch_size
idx = 0 
index_list = [0]
patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices_cao = generate_patches([x_cao[idx][:,:,0]],
                                                                                                                                            [masks_cao[idx]],
                                                                                                                                            [lon_lats_cao[idx]],
                                                                                                                                            max_vals,
                                                                                                                                            min_vals,
                                                                                                                                            autoencoder_predict,
                                                                                                                                            strides=[1, strides, strides,1])
# patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices_cao = generate_patches([x[:,:,0] for x in x_cao],
#                                                                                                                                             masks_cao,
#                                                                                                                                             lon_lats_cao,
#                                                                                                                                             max_vals,
#                                                                                                                                             min_vals,
#                                                                                                                                             autoencoder_predict,
#                                                                                                                                             strides=[1, strides, strides,1])

encoder_128 = load_model("/uio/hume/student-u37/fslippe/data/models/patch_size128/filter128/encoder_dnb_l95_z50_ps128_f128_1e3_201812-202312.h5")
encoded_patches_128_cao = encoder_128.predict(patches_cao)
encoded_patches_flat_128_cao = encoded_patches_128_cao.reshape(encoded_patches_128_cao.shape[0], -1)
cluster = joblib.load("/uio/hume/student-u37/fslippe/data/models/cluster_dnb_l95_z50_ps128_band29_filter128_K12.pkl" )
labels = cluster.predict(encoded_patches_flat_128_cao)
label_1 = 8
label_2 = 10

global_min = np.min([np.min(cm) for cm in cluster.labels_])
global_max = np.max([np.max(cm) for cm in cluster.labels_])+1

# encoder_64 = load_model("/uio/hume/student-u37/fslippe/data/models/patch_size128/filter64/encoder_dnb_l95_z50_ps128_f64_201812-202312.h5")
# encoded_patches_64_cao = encoder_64.predict(patches_cao)
# encoded_patches_flat_64_cao = encoded_patches_64_cao.reshape(encoded_patches_64_cao.shape[0], -1)
# cluster = joblib.load("/uio/hume/student-u37/fslippe/data/models/cluster_dnb_l95_z50_ps128_band29.pkl" )
# labels_64 = cluster.predict(encoded_patches_flat_64_cao)

label_map, lon_map, lat_map = process_label_maps(labels,all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, indices_cao, global_max, n_patches_tot_cao, patch_size, strides, label_1, label_2, size_thr_1=20, size_thr_2=20)

# Example usage of the function
extent = [-15, 25, 58, 84]
plot_filtered_map(label_map, lon_map, lat_map, idx, extent, global_max, dates)
plt.show()

for i in index_list:
    valid_lons, valid_lats = get_valid_lons_lats(x_cao[i][:,:,0],
                                                lon_lats_cao[i],
                                                label_map[i],
                                                lon_map[i],
                                                lat_map[i],
                                                dates[i],
                                                times[i],
                                                open_label=label_1,
                                                closed_label=label_2,
                                                p_level=950,
                                                angle_thr=5,
                                                size_threshold_1=None,
                                                size_threshold_2=None,
                                                plot=False,
                                                extent= [-15, 25, 58, 84])


model_boundaries, model_areas = process_model_masks(index_list, lon_map, lat_map, valid_lons, valid_lats, indices_cao, label_map, label_1, label_2, plot=True)

filepath = "/uio/hume/student-u37/fslippe/data/labeled_data/results_backup_20230105"
with open(filepath, "r") as f:
    data = json.load(f)["data"]["image_results"]

labeled_data = pd.json_normalize(data)
dates_cao = dates
mod_min_cao = times

labeled_areas, labeled_boundaries = get_area_and_border_mask(x_cao, dates, times, masks_cao, labeled_data, dates_cao, mod_min_cao, reduction=strides, plot=True)
# fig, axs = plt.subplots(1,2, figsize=[10,10])
# cb = axs[0].imshow(labeled_areas[0])
# plt.colorbar(cb)
# cb = axs[1].imshow(labeled_boundaries[0])
# plt.colorbar(cb)


area_scores, border_scores = calculate_scores_and_plot(model_boundaries, model_areas, labeled_boundaries, labeled_areas, plot=True)


