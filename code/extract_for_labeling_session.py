# %%
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
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

# %%
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
max_vals = np.load("%smodels/max_vals_dnb_l95_z50_ps128_(29)_cao_months_202012-202111.npy" %data_loc)


# %%
from autoencoder import SobelFilterLayer, SimpleAutoencoder
print(len(bands))

#encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
encoder = load_model("%smodels/encoder_scheduler_250k_dnb_l90_z50_fcao_months_202012_(29)_202111-64" %data_loc)

# %%
### EXTRACT CAO AND NOn CAO CASES
import importlib
import extract_training_data
importlib.reload(extract_training_data)
from extract_training_data import *
patch_size = 128

start = "20190101"
end = "20190430"
dates = generate_date_list(start, end)
start = "20191001"
end = "20191231"
year_chosen = start[:4]
dates.extend(generate_date_list(start, end))
#x_cao, dates_cao, masks_cao = extract_1km_data("/scratch/fslippe/modis/MOD02/july_2021/", bands=bands, start_date=start_converted, end_date=end_converted)
bands = [29]
# x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min = extract_1km_data("/scratch/fslippe/modis/MOD02/cao_test_data/",
#                                                            bands=bands,
#                                                          start_date=start_converted,
#                                                          end_date=end_converted,
#                                                          return_lon_lat=True)
folder = "/scratch/fslippe/modis/MOD02/2019/ /scratch/fslippe/modis/MOD02/2020/ /scratch/fslippe/modis/MOD02/2021/ /scratch/fslippe/modis/MOD02/2022/ /scratch/fslippe/modis/MOD02/2023/"
#folder = "/scratch/fslippe/modis/MOD02/2020/"
x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = extract_1km_data(folder,
                                                         bands=bands,
                                                         #start_date=start_converted,
                                                         #end_date=end_converted,
                                                         date_list=dates,
                                                         return_lon_lat=True,
                                                         data_loc=data_loc,
                                                         data_type="npy",
                                                         combine_pics=True)

x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
x_cao = list(x_cao)
dates_cao = list(dates_cao)
len(x_cao)


# %%
#### EXTRACTING AND ENCODING PATCHES + SAVING indexes of belonging files
autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
stride = patch_size
patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices_cao = generate_patches([x[:,:,0] for x in x_cao],
                                                                                                                                       masks_cao,
                                                                                                                                       lon_lats_cao,
                                                                                                                                       max_vals,
                                                                                                                                       autoencoder_predict,
                                                                                                                                       strides=[1, stride, stride,1])
n_patches = len(patches_cao)

# %%
n_chunks = 10
chunk_size = n_patches // n_chunks

patch_chunks = [patches_cao[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]

# Handle the remaining patches for the case where n_patches is not exactly divisible by n_chunks
if n_patches % n_chunks != 0:
    patch_chunks.append(patches_cao[n_chunks * chunk_size:])

encoded_patch_chunks = [encoder.predict(chunk) for chunk in patch_chunks]
encoded_patches_cao = np.concatenate(encoded_patch_chunks)

encoded_patches_flat_cao = encoded_patches_cao.reshape(encoded_patches_cao.shape[0], -1)
gc.collect()


# %%
cluster = joblib.load("/uio/hume/student-u37/fslippe/data/models/cluster_dnb_l90_z50_ps128_f64_(29)_%s-%s.pkl" %("cao_months_202012", "202111"))
global_min = np.min([np.min(cm) for cm in cluster.labels_])
global_max = np.max([np.max(cm) for cm in cluster.labels_])+1
print(global_max)
labels = cluster.predict(encoded_patches_flat_cao)

# %%
import plot_functions 
importlib.reload(plot_functions)
from plot_functions import * 
size_threshold = 30
less_than = False
save_img_with_labels(x_cao, lon_lats_cao, n_patches_tot_cao,
                      indices_cao,
                      labels,
                      starts_cao,
                      ends_cao,  
                      shapes_cao,
                      dates_cao,
                      mod_min_cao,
                      desired_label = [3, 0],
                      size_threshold = size_threshold,
                      less_than=less_than,
                      patch_size = patch_size,
                      global_max = global_max,
                      max_pics =50,
                      shuffle=True,
                      plot=False,
                      save_np="%scao_thr%s_%s" %("no" if less_than else "", size_threshold, year_chosen))
size_threshold = 35
less_than = False
save_img_with_labels(x_cao, lon_lats_cao, n_patches_tot_cao,
                      indices_cao,
                      labels,
                      starts_cao,
                      ends_cao,  
                      shapes_cao,
                      dates_cao,
                      mod_min_cao,
                      desired_label = [3, 0],
                      size_threshold = size_threshold,
                      less_than=less_than,
                      patch_size = patch_size,
                      global_max = global_max,
                      max_pics =50,
                      shuffle=True,
                      plot=False,
                      save_np="%scao_thr%s_%s" %("no" if less_than else "", size_threshold, year_chosen))            

size_threshold = 40
less_than = False
save_img_with_labels(x_cao, lon_lats_cao, n_patches_tot_cao,
                      indices_cao,
                      labels,
                      starts_cao,
                      ends_cao,  
                      shapes_cao,
                      dates_cao,
                      mod_min_cao,
                      desired_label = [3, 0],
                      size_threshold = size_threshold,
                      less_than=less_than,
                      patch_size = patch_size,
                      global_max = global_max,
                      max_pics =50,
                      shuffle=True,
                      plot=False,
                      save_np="%scao_thr%s_%s" %("no" if less_than else "", size_threshold, year_chosen))           
                      
size_threshold = 2
less_than = True
save_img_with_labels(x_cao, lon_lats_cao, n_patches_tot_cao,
                      indices_cao,
                      labels,
                      starts_cao,
                      ends_cao,  
                      shapes_cao,
                      dates_cao,
                      mod_min_cao,
                      desired_label = [3, 0],
                      size_threshold = size_threshold,
                      less_than=less_than,
                      patch_size = patch_size,
                      global_max = global_max,
                      max_pics =50,
                      shuffle=True,
                      plot=False,
                      save_np="%scao_thr%s_%s" %("no" if less_than else "", size_threshold, year_chosen))     
size_threshold = 1
less_than = True
save_img_with_labels(x_cao, lon_lats_cao, n_patches_tot_cao,
                      indices_cao,
                      labels,
                      starts_cao,
                      ends_cao,  
                      shapes_cao,
                      dates_cao,
                      mod_min_cao,
                      desired_label = [3, 0],
                      size_threshold = size_threshold,
                      less_than=less_than,
                      patch_size = patch_size,
                      global_max = global_max,
                      max_pics =50,
                      shuffle=True,
                      plot=False,
                      save_np="%scao_thr%s_%s" %("no" if less_than else "", size_threshold, year_chosen))     
size_threshold = 0
less_than = True
save_img_with_labels(x_cao, lon_lats_cao, n_patches_tot_cao,
                      indices_cao,
                      labels,
                      starts_cao,
                      ends_cao,  
                      shapes_cao,
                      dates_cao,
                      mod_min_cao,
                      desired_label = [3, 0],
                      size_threshold = size_threshold,
                      less_than=less_than,
                      patch_size = patch_size,
                      global_max = global_max,
                      max_pics =50,
                      shuffle=True,
                      plot=False,
                      save_np="%scao_thr%s_%s" %("no" if less_than else "", size_threshold, year_chosen))     


