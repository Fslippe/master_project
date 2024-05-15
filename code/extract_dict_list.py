
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
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow import keras    
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN
from pyhdf.SD import SD, SDC
import matplotlib as mpl
#tf.config.threading.set_inter_op_parallelism_threads(1)
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from pyhdf.error import HDF4Error
from functions import *
from extract_training_data import *
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
import reanalysis_functions 
importlib.reload(reanalysis_functions)
from reanalysis_functions import * 
from calculate_scores import * 


# Visualize the result
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,"  , len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#bands = [6, 7, 20, 28, 28, 31]
bands=[29]
#bands=[1]
max_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/max_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
min_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/min_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
merra_folder = "/uio/hume/student-u37/fslippe/MERRA/"


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
last_filter = 128
threshold = 5
n_Ks = [13]
years = [2019, 2020, 2021, 2022, 2023]#, 2021, 2022, 2023]
size_thresholds = [0, 50, 100, 300] #[300, 0, 100]#, 100, 300]
folder = "/scratch/fslippe/modis/MOD02_npz/2019/ /scratch/fslippe/modis/MOD02_npz/2020/ /scratch/fslippe/modis/MOD02_npz/2021/ /scratch/fslippe/modis/MOD02_npz/2022/ /scratch/fslippe/modis/MOD02_npz/2023/"

def extract_for_nKs(years, n_Ks, size_thresholds, threshold):
    for n_K in n_Ks:
        open_label = np.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_opencell_label.npy")
        closed_label = np.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_closedcell_label.npy")
        
        times_folder = f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cao_date_time_lists/n_K_{n_K}/"

        dates_cao = []
        times_cao = []
        for yr in years:
            time_dict = np.load(times_folder + f"times_patch_size{patch_size}_filter{last_filter}_nK{n_K}_thr{threshold}_{yr}.npy", allow_pickle=True).item()
            half_step = len(time_dict["dates"]) // 2

            dates_cao.extend(time_dict["dates"][half_step:])
            times_cao.extend(time_dict["times"][half_step:])

        x, dates, masks, lon_lats, mod_min = extract_1km_data(folder,
                                                            bands=bands,
                                                            date_list=np.unique(dates_cao),
                                                            return_lon_lat=True,
                                                            data_loc=data_loc,
                                                            data_type="npz",
                                                            combine_pics=True)

        x, dates, masks, lon_lats, mod_min = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x, dates, masks, lon_lats, mod_min) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
        x = list(x)
        dates = list(dates)
        len(x)

        print("unique dates", len(np.unique(dates_cao)))
        print("Total images", len((dates_cao)))


        indices = [index for index, (date, mod_min) in enumerate(zip(dates, mod_min)) if (date, mod_min) in zip(dates_cao, times_cao)]
        x_cao = [x[index] for index in indices]
        dates_cao = [dates[index] for index in indices]
        masks_cao = [masks[index] for index in indices]
        lon_lats_cao = [lon_lats[index] for index in indices]
        mod_min_cao = [mod_min[index] for index in indices]

        del x
        del dates
        del masks
        del lon_lats
        del mod_min 
        gc.collect()


        autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
        stride = 16
        print("GENERATING PATCHES")
        patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices = generate_patches([x[:,:,0] for x in x_cao],
                                                                                                            masks_cao,
                                                                                                            lon_lats_cao,
                                                                                                            max_vals,
                                                                                                            min_vals,
                                                                                                            autoencoder_predict,
                                                                                                            strides=[1, stride, stride,1])                                                                                                   
        gc.collect()
        
        with tf.device('/CPU:0'):   
            encoded_patches_flat_cao = load_and_predict_encoder(patch_size, last_filter, patches)

        del patches 
        gc.collect()


        lon = xr.open_dataset(f"{merra_folder}2020/MERRA2_400.tavg1_2d_slv_Nx.20200312.SUB.nc").lon.values
        lat = xr.open_dataset(f"{merra_folder}2020/MERRA2_400.tavg1_2d_slv_Nx.20200312.SUB.nc").lat.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)


        pat_lon = []
        pat_lat = []
        for i in range(len(all_lon_patches)):
            if all_lon_patches[i].ndim == 2:
                plat = np.mean(np.expand_dims(all_lat_patches[i], axis=0), axis=(1,2))
                plon = np.mean(np.expand_dims(all_lon_patches[i], axis=0), axis=(1,2))

            else:
                plon = np.mean(all_lon_patches[i], axis=(1, 2))
                plat = np.mean(all_lat_patches[i], axis=(1, 2))
            pat_lon.append(plon)
            pat_lat.append(plat)

        pat_lon = np.concatenate(np.array(pat_lon), axis=0)
        pat_lat = np.concatenate(np.array(pat_lat), axis=0)    
        labels, global_min, global_max = get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K)

        del encoded_patches_flat_cao 
        gc.collect()
        
        for size_threshold in size_thresholds:
            label_map, lon_map, lat_map = process_label_maps(labels,
                                                    all_lon_patches,
                                                    all_lat_patches,
                                                    starts,                 
                                                    ends,
                                                    shapes,         
                                                    indices,
                                                    n_K,             
                                                    n_patches_tot,              
                                                    patch_size,
                                                    stride,
                                                    closed_label, 
                                                    open_label, 
                                                    size_thr_1=size_threshold if size_threshold != 0 else None, 
                                                    size_thr_2=size_threshold if size_threshold != 0 else None)

            dict_list = []

            # Setup arguments for parallel processing
            args = []
            for i in range(len(x_cao)):
                args.append((i, dates_cao[i], mod_min_cao[i], lon_map[i], lat_map[i], label_map[i], closed_label, open_label, lon_mesh, lat_mesh))

            # Perform parallel processing
            with ProcessPoolExecutor(max_workers=len(x_cao) if len(x_cao) < 128 else 128) as executor:
                dict_list = list(executor.map(process_lon_lat_wind_check, args))

            # Save results to file
            if len(years) > 1:
                np.save(f"/uio/hume/student-u37/fslippe/data/model_pred_info/filter{last_filter}/dict_filter{last_filter}_nK{n_K}_caothr{threshold}_sizethr_{size_threshold}_stride{stride}_{years[0]}-{years[-1]}", dict_list)
            else:
                file_path = f"/uio/hume/student-u37/fslippe/data/model_pred_info/filter{last_filter}/dict_filter{last_filter}_nK{n_K}_caothr{threshold}_sizethr_{size_threshold}_stride{stride}_{years[0]}"
                count = 1
                new_file_path = file_path

                # Check if the file already exists, and if it does, modify the file name
                while os.path.exists(new_file_path + ".npy"):
                    new_file_path = file_path + f"_{count}"
                    count += 1

                # Save the data to the new file name
                np.save(new_file_path, dict_list)
            
        
        del x_cao 
        del dates_cao 
        del masks_cao 
        del lon_lats_cao 
        del mod_min_cao 
        gc.collect()

if threshold >= 30:
    extract_for_nKs(years, n_Ks, size_thresholds, threshold)
else:
    for year in years:
        extract_for_nKs([year], n_Ks, size_thresholds, threshold)
