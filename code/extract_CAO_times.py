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
import sys
if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = input("Enter the year: ")

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
from autoencoder import SobelFilterLayer, SimpleAutoencoder
print(len(bands))

#encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
max_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/max_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
min_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/min_val_dnb_l95_z50_ps128_band29_2018-2023.npy")



### EXTRACT CAO AND NOn CAO CASES
import importlib
import extract_training_data
importlib.reload(extract_training_data)
from extract_training_data import *
patch_size = 128

filters = [128]
thresholds = [5, 10, 15, 30 ]#[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_Ks = [13]#[10, 11, 12, 13, 14, 15, 16]

year = 2023
# start = f"{year}0601"
# end = f"{year}0731"
# dates = generate_date_list(start, end)
off_season = False
start = f"{year}0101"
end = f"{year}0430"
dates = generate_date_list(start, end)

start = f"{year}1001"
end = f"{year}1231"
dates.extend(generate_date_list(start, end))


folder = "/scratch/fslippe/modis/MOD02_npz/2019/ /scratch/fslippe/modis/MOD02_npz/2020/ /scratch/fslippe/modis/MOD02_npz/2021/ /scratch/fslippe/modis/MOD02_npz/2022/ /scratch/fslippe/modis/MOD02_npz/2023/"
x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = extract_1km_data(folder,
                                                         bands=bands,
                                                         date_list=dates,
                                                         return_lon_lat=True,
                                                         data_loc=data_loc,
                                                         data_type="npz",
                                                         combine_pics=True)

x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
x_cao = list(x_cao)
dates_cao = list(dates_cao)
len(x_cao)



autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices = generate_patches([x[:,:,0] for x in x_cao],
                                                                                                    masks_cao,
                                                                                                    lon_lats_cao,
                                                                                                    max_vals,
                                                                                                    min_vals,
                                                                                                    autoencoder_predict,
                                                                                                    strides=[1, patch_size, patch_size,1])

                                                                                                    
del lon_lats_cao,
gc.collect()

for filter in filters:

    if filter == 32:
        encoder = load_model("/uio/hume/student-u37/fslippe/data/models/patch_size128/filter32/encoder_dnb_l95_z50_ps128_f32_1e3_201812-202312.h5")
    elif filter == 64:
        encoder = load_model("/uio/hume/student-u37/fslippe/data/models/patch_size128/filter64/encoder_dnb_l95_z50_ps128_f64_1e3_201812-202312_epoch_500.h5")
    elif filter == 128:
        encoder = load_model("/uio/hume/student-u37/fslippe/data/models/patch_size128/filter128/encoder_dnb_l95_z50_ps128_f128_1e3_201812-202312.h5")

    n_chunks = 10
    n_patches = len(patches)
    chunk_size = n_patches // n_chunks

    patch_chunks = [patches[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]

    # Handle the remaining patches for the case where n_patches is not exactly divisible by n_chunks
    if n_patches % n_chunks != 0:
        patch_chunks.append(patches[n_chunks * chunk_size:])

    encoded_patch_chunks = [encoder.predict(chunk) for chunk in patch_chunks]
    encoded_patches = np.concatenate(encoded_patch_chunks)

    encoded_patches_flat = encoded_patches.reshape(encoded_patches.shape[0], -1)
    gc.collect()

    import plot_functions
    importlib.reload(plot_functions)
    from plot_functions import * 

    for n_K in n_Ks:
        cluster =  joblib.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size128/filter{filter}/clustering/cluster_dnb_l95_z50_ps128_band29_filter{filter}_K{n_K}.pkl")
        label_1 = int(np.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size128/filter{filter}/clustering/cluster_dnb_l95_z50_ps128_band29_filter{filter}_K{n_K}_opencell_label.npy"))
        label_2 = int(np.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size128/filter{filter}/clustering/cluster_dnb_l95_z50_ps128_band29_filter{filter}_K{n_K}_closedcell_label.npy"))
        print(label_1, label_2)

        labels = cluster.predict(encoded_patches_flat)
        folder = f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{filter}/clustering/cao_date_time_lists/n_K_{n_K}/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for threshold in thresholds:
            dates, times = save_img_with_labels(x_cao,
                                n_patches_tot,
                                indices,
                                labels,
                                starts,
                                ends,  
                                shapes,
                                dates_cao,
                                mod_min_cao,
                                plot=False,
                                less_than=False,
                                max_pics = 10000,
                                desired_label=[label_1,label_2],
                                size_threshold=threshold,
                                patch_size=patch_size,
                                global_max=n_K)
            print(len(dates))
            time_dict = {"dates": dates, "times": times}
            if off_season:
                import os

                file_name = f"times_patch_size{patch_size}_filter{filter}_nK{n_K}_thr{threshold}_{year}"
                if off_season:
                    file_name += "_off_season"
                file_name += ".npz"

                # Generate a unique file name by appending a number if the file already exists
                count = 1
                while True:
                    new_file_name = file_name if count == 1 else f"{file_name}_{count}"
                    new_file_path = os.path.join(folder, new_file_name)
                    
                    if not os.path.exists(new_file_path):
                        break
                    count += 1

                # Save the data to the new file
                np.save(new_file_path, time_dict)
            else:
                np.save(folder + f"times_patch_size{patch_size}_filter{filter}_nK{n_K}_thr{threshold}_{year}%s" %("_off_season" if off_season else ""), time_dict)


    gc.collect()
        
