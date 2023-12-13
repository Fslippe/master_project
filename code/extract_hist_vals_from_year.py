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
from autoencoder import SobelFilterLayer, SimpleAutoencoder


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
print(len(bands))

#encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
encoder = load_model("%smodels/encoder_scheduler_250k_dnb_l90_z50_fcao_months_202012_(29)_202111-64" %data_loc)

### EXTRACT CAO AND NOn CAO CASES
patch_size = 128
start = "20190101"
end = "20190430"
dates = generate_date_list(start, end)
start = "20191001"
end = "20191231"
dates.extend(generate_date_list(start, end))
target_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
size_threshold = 4
desired_label = [3, 0]
grid_resolution = 128e3
filters = 64
year = dates_cao[0][:4]

def extract_files_and_patches():
    folder = "/scratch/fslippe/modis/MOD02/2019/ /scratch/fslippe/modis/MOD02/2020/ /scratch/fslippe/modis/MOD02/2021/ /scratch/fslippe/modis/MOD02/2022/ /scratch/fslippe/modis/MOD02/2023/"
    x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = extract_1km_data(folder,
                                                            bands=bands,
                                                            date_list=dates,
                                                            return_lon_lat=True,
                                                            data_loc=data_loc,
                                                            data_type="npy",
                                                            combine_pics=True)

    x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
    x_cao = list(x_cao)
    dates_cao = list(dates_cao)
    gc.collect()
        
    autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)

    patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices = generate_patches([xi[:,:,0] for xi in x_cao],
                                                                                                                masks_cao,
                                                                                                                lon_lats_cao,
                                                                                                                max_vals,
                                                                                                                autoencoder_predict,
                                                                                                                strides=[1, patch_size, patch_size,1])
    n_patches = len(patches)



def encode_and_cluster():
    cluster = joblib.load("/uio/hume/student-u37/fslippe/data/models/cluster_dnb_l90_z50_ps128_f64_(29)_%s-%s.pkl" %("cao_months_202012", "202111"))
    encoder = load_model("/uio/hume/student-u37/fslippe/data/models/encoder_scheduler_250k_dnb_l90_z50_fcao_months_202012_(29)_202111-64")

    n_chunks = 10
    chunk_size = n_patches // n_chunks
    patch_chunks = [patches[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]

    # Handle the remaining patches for the case where n_patches is not exactly divisible by n_chunks
    if n_patches % n_chunks != 0:
        patch_chunks.append(patches[n_chunks * chunk_size:])

    encoded_patch_chunks = [encoder.predict(chunk) for chunk in patch_chunks]
    encoded_patches = np.concatenate(encoded_patch_chunks)

    encoded_patches_flat = encoded_patches.reshape(encoded_patches.shape[0], -1)
    gc.collect()
    labels = cluster.predict(encoded_patches_flat)



def find_hist_vals():
    for target_month in target_months:
        print("/uio/hume/student-u37/fslippe/data/hist_maps/hist_counts_res%s_thr%s_ps%s_filters%s_%s%s" %(int(grid_resolution*1e-3), size_threshold, patch_size, filters, year, str(target_month).zfill(2)))
        idx = np.array([i for i, date_str in enumerate(dates_cao) if datetime.datetime.strptime(date_str, '%Y%j').month == target_month])
        x_grid, y_grid, counts = generate_hist_map(
                                    [n_patches_tot[i] for i in idx],
                                    [indices[i] for i in idx],
                                    labels,
                                    [starts[i] for i in idx],
                                    [ends[i] for i in idx],  
                                    [shapes[i] for i in idx],
                                    [all_lon_patches[i] for i in idx],
                                    [all_lat_patches[i] for i in idx],
                                    [dates_cao[i] for i in idx],
                                    desired_label=desired_label,
                                    size_threshold=size_threshold,
                                    patch_size=patch_size,
                                    global_max=global_max,
                                    projection=projection,
                                    grid_resolution=grid_resolution)
        tot_days = len(set(np.array(dates_cao)[idx]))
        ds = {"days": tot_days, "counts": counts}
        np.save("/uio/hume/student-u37/fslippe/data/hist_maps/hist_counts_res%s_thr%s_ps%s_filters%s_%s%s" %(grid_resolution, size_threshold, patch_size, filters, year, str(target_month).zfill(2)), ds)
