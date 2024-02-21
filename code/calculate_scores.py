import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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


def import_label_data(label_data_file_path):
    folder_loc = "/uio/hume/student-u37/fslippe/labeling_session/npy_files/"
    dates_block = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_block.npy")
    times_block = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_block.npy")
    # dates_rest = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_rest.npy")
    # times_rest = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_rest.npy")
    # dates = np.append(dates_block, dates_rest)
    # times = np.append(times_block, times_rest)
    dates = dates_block
    times = times_block

    x_cao = []
    masks_cao = []
    lon_lats_cao = []

    #dates, times = dates_block[10:12], times_block[10:12]
    s=0

    for (d, m) in zip(dates, times):
        s+=1
        arr = np.load(f"{folder_loc}MOD021KM.A%s.%s.combined.npy" %(d, m))
        x_cao.append(arr)
        arr = np.load(f"{folder_loc}masks/masks.A%s.%s.combined.npy" %(d, m))
        masks_cao.append(arr)
        arr = np.load(f"{folder_loc}lon_lats/lon_lats.A%s.%s.combined.npy" %(d, m))
        lon_lats_cao.append(arr)
        #print("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A%s.%s_combined" %(d, m))
        #idx = np.where((np.array(dates_cao) == d) & (np.array(mod_min_cao) == m))[0][0]
        #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A%s.%s.combined.npy" %(d, m), masks_cao[idx])
        #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A%s.%s.combined.npy" %(d, m), lon_lats_cao[idx])
        #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/new_files/MOD021KM.A%s.%s.combined" %(d, m), arr)
        
    max_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/max_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
    min_vals = np.load("/uio/hume/student-u37/fslippe/data/models/patch_size128/min_val_dnb_l95_z50_ps128_band29_2018-2023.npy")

    with open(label_data_file_path, "r") as f:
        data = json.load(f)["data"]["image_results"]

    labeled_data = pd.json_normalize(data)
    
    return dates, times, labeled_data, x_cao, masks_cao, lon_lats_cao , max_vals, min_vals 


def load_and_predict_encoder(patch_size, last_filter, patches_cao):
    if last_filter == 128:
        encoder = load_model(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter128/encoder_dnb_l95_z50_ps128_f128_1e3_201812-202312.h5")
    elif last_filter == 64:
        encoder = load_model(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter64/encoder_dnb_l95_z50_ps128_f64_1e3_201812-202312_epoch_500.h5")
    elif last_filter == 32:
        encoder = load_model(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter32/encoder_dnb_l95_z50_ps128_f32_1e3_201812-202312.h5")

    encoded_patches_cao = encoder.predict(patches_cao)
    encoded_patches_flat_cao = encoded_patches_cao.reshape(encoded_patches_cao.shape[0], -1)

    return encoded_patches_flat_cao


def get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K):
    print("cluster load loc:", "/uio/hume/student-u37/fslippe/data/models/patch_size%s/filter%s/clustering/cluster_dnb_l95_z50_ps128_band29_filter%s_K%s.pkl"  %(patch_size, last_filter, last_filter, n_K))
    cluster = joblib.load("/uio/hume/student-u37/fslippe/data/models/patch_size%s/filter%s/clustering/cluster_dnb_l95_z50_ps128_band29_filter%s_K%s.pkl" %(patch_size, last_filter, last_filter, n_K))
    labels = cluster.predict(encoded_patches_flat_cao)

    global_min = 0
    global_max = n_K 
    return labels, global_min, global_max

def manually_find_cloud_labels(min_vals, max_vals, autoencoder_predict, patch_size, last_filter, n_K):
    x_test1 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/radiance_2021080_1120_combined.npy")
    x_test2 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/radiance_2023062_1100_combined.npy")
    x_test3 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/radiance_2023065_1125_combined.npy")

    x_test4 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A2019060.1030.combined.npy")
    x_test5 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A2022347.1150.combined.npy")
    x_test6 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A2022120.955.combined.npy")
    x_test = ([x_test1, x_test2, x_test3, x_test4, x_test5, x_test6])


    masks_test1 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/mask_2021080_1120_combined.npy")
    masks_test2 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/mask_2023062_1100_combined.npy")
    masks_test3 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/mask_2023065_1125_combined.npy")

    masks_test4 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A2019060.1030.combined.npy")
    masks_test5 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A2022347.1150.combined.npy")
    masks_test6 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A2022120.955.combined.npy")
    masks_test = ([masks_test1, masks_test2, masks_test3, masks_test4, masks_test5, masks_test6])
    

    lon_lats_test1 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/lonlat_2021080_1120_combined.npy")
    lon_lats_test2 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/lonlat_2023062_1100_combined.npy")
    lon_lats_test3 = np.load("/uio/hume/student-u37/fslippe/data/cao_examples/lonlat_2023065_1125_combined.npy")

    lon_lats_test4 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A2019060.1030.combined.npy")
    lon_lats_test5 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A2022347.1150.combined.npy")
    lon_lats_test6 = np.load("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A2022120.955.combined.npy")
    lon_lats_test = ([lon_lats_test1, lon_lats_test2, lon_lats_test3, lon_lats_test4, lon_lats_test5, lon_lats_test6])

    patches_test, all_lon_patches_test, all_lat_patches_test, starts_test, ends_test, shapes_test, n_patches_tot_test, indices_test = generate_patches([x[:,:,0] for x in x_test],
                                                                                                                                            masks_test,
                                                                                                                                            lon_lats_test,
                                                                                                                                            max_vals,
                                                                                                                                            min_vals,
                                                                                                                                            autoencoder_predict,
                                                                                                                                            strides=[1, patch_size, patch_size,1])
    encoded_patches_flat_cao = load_and_predict_encoder(patch_size, last_filter, patches_test)
    labels, global_min, global_max = get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K)
    plot_img_cluster_mask(x_test,
                      labels,#, labels_64],
                      masks_test,
                      starts_test,
                      ends_test,
                      shapes_test,
                      indices_test,
                      ["1", "2", "3", "4", "5", "6"],
                      n_patches_tot_test,
                      patch_size,
                      global_min,
                      global_max,
                      index_list=[0,1,2,3,4,5],
                      chosen_label=3,
                      one_fig=True,
                      save=None)

    plt.ion()
                 
    plt.show()
    open_label = int(input("Open cell label: "))
    # Prompt for closed cell label
    closed_label = int(input("Closed cell label: "))
    plt.ioff()
    plt.close()
    np.save(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_opencell_label", open_label)
    np.save(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_closedcell_label", closed_label)



def main():
    bands=[29]
    patch_size = 128
    last_filter = 64
    strides = 128    #patch_size
    idx = 0 
    index_list = [21, 0]
    size_threshold = 10
    dates, times, labeled_data, x_cao, masks_cao, lon_lats_cao , max_vals, min_vals  = import_label_data("/uio/hume/student-u37/fslippe/data/labeled_data/results_backup_20240118")  
    autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
    
    patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices_cao = generate_patches([x[:,:,0] for x in x_cao],
                                                                                                                                                masks_cao,
                                                                                                                                                lon_lats_cao,
                                                                                                                                                max_vals,
                                                                                                                                                min_vals,
                                                                                                                                                autoencoder_predict,
                                                                                                                                                strides=[1, strides, strides,1])

    encoded_patches_flat_cao = load_and_predict_encoder(patch_size, last_filter, patches_cao)

    for n_K in [10, 11]:
        if not os.path.exists(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_opencell_label.npy"):
            manually_find_cloud_labels(min_vals, max_vals, autoencoder_predict, patch_size, last_filter, n_K)
        if not os.path.exists(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_closedcell_label.npy"):
            manually_find_cloud_labels(min_vals, max_vals, autoencoder_predict, patch_size, last_filter, n_K)
    
        open_label = np.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_opencell_label.npy")
        closed_label = np.load(f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_closedcell_label.npy")
        

        labels, global_min, global_max = get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K)
        label_map, lon_map, lat_map = process_label_maps(labels,
                                                        all_lon_patches_cao,
                                                        all_lat_patches_cao,
                                                        starts_cao,
                                                        ends_cao,
                                                        shapes_cao,
                                                        indices_cao,
                                                        global_max,
                                                        n_patches_tot_cao,
                                                        patch_size,
                                                        strides,
                                                        closed_label, 
                                                        open_label, 
                                                        size_thr_1=size_threshold, 
                                                        size_thr_2=size_threshold)

            # Example usage of the function
        extent = [-15, 25, 58, 84]
        # plot_filtered_map(label_map, lon_map, lat_map, idx, extent, global_max, dates)
        # plt.show()

        for i in index_list:
            valid_lons, valid_lats = get_valid_lons_lats(x_cao[i][:,:,0],
                                                        lon_lats_cao[i],
                                                        label_map[i],
                                                        lon_map[i],
                                                        lat_map[i],
                                                        dates[i],
                                                        times[i],
                                                        open_label=closed_label,
                                                        closed_label=open_label,
                                                        p_level=950,
                                                        angle_thr=5,
                                                        size_threshold_1=None,
                                                        size_threshold_2=None,
                                                        plot=False,
                                                        extent= [-15, 25, 58, 84])


        model_boundaries, model_areas = process_model_masks(index_list, lon_map, lat_map, valid_lons, valid_lats, indices_cao, label_map, closed_label, open_label, plot=False)

        labeled_areas, labeled_boundaries = get_area_and_border_mask(x_cao, dates, times, masks_cao, labeled_data, reduction=strides, index_list=index_list, plot=False)
        plt.figure()
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(model_areas[0])
        axs[1].imshow(labeled_areas[0])
        plt.show()
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(model_areas[1])
        axs[1].imshow(labeled_areas[1])
        plt.show()
        # fig, axs = plt.subplots(1,2, figsize=[10,10])
        # cb = axs[0].imshow(labeled_areas[0])
        # plt.colorbar(cb)
        # cb = axs[1].imshow(labeled_boundaries[0])
        # plt.colorbar(cb)


        area_scores, border_scores, weighted_area_scores, weighted_border_scores = calculate_scores_and_plot(model_boundaries, model_areas, labeled_boundaries, labeled_areas, plot=False)

        folder = f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{last_filter}/clustering/scores/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        np.save(folder + f"area_scores_cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_res{strides}_thr{size_threshold}", area_scores)
        np.save(folder + f"border_scores_cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_res{strides}_thr{size_threshold}", border_scores)
        np.save(folder + f"weighted_area_scores_cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_res{strides}_thr{size_threshold}", weighted_area_scores)
        np.save(folder + f"weighted_border_scores_cluster_dnb_l95_z50_ps{patch_size}_band29_filter{last_filter}_K{n_K}_res{strides}_thr{size_threshold}", weighted_border_scores)


if __name__ == "__main__":
    main()


