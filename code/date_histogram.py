import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
from keras.layers import Input, Dense, Flatten, Reshape
from sklearn.feature_extraction import image as sk_image
from concurrent.futures import ProcessPoolExecutor
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


#### LOAD MODELS 
cluster = joblib.load('/uio/hume/student-u37/fslippe/data/models/cluster_winter_2020_21_dnb_landmask_band(29)_lab1.pkl')
encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_band(29)_filter_encoder")
#encoder =  load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_band(29)_filter_encoder")
max_vals = np.load("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_band(29)_max_vals.npy")


#### SETUP + LOAD DATA
bands=[29]
patch_size = 64
desired_label = 2
size_threshold = 10  # Adjust based on the minimum size of the region you are interested in
print(len(bands))
autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
folder = "/scratch/fslippe/modis/MOD02/night_1km/ /scratch/fslippe/modis/MOD02/may-nov_2021 /scratch/fslippe/modis/MOD02/daytime_1km/ /scratch/fslippe/modis/MOD02/boundary_1km/"

start_date = "20201201"
end_date = "20210131"

start_date = "20210201"
end_date = "20210331"

start_date = "20210401"
end_date = "20210801"


start_date_list = ["20210801"]#, "20211001"]#["20201201", "20210201", "20210401"]
end_date_list = ["20211130"]#, "20211130"]#["20210131", "20210331", "20210801"]


def get_date_hist(cluster, encoder, max_vals):
    for (start_date, end_date) in zip(start_date_list, end_date_list):
        gc.collect()
        dates_converted = []
        start_converted = convert_to_day_of_year(start_date)
        end_converted = convert_to_day_of_year(end_date)
        print(start_converted)
        print(end_converted)

        x, dates, masks = extract_1km_data(folder, bands=bands, start_date=start_converted, end_date=end_converted)
        x, dates, masks = zip(*[(xi, date, mask) for xi, date, mask in zip(x, dates, masks) if (xi.shape[0] > 64) and (xi.shape[1] > 64)])
        x = list(x)
        dates = list(dates)


        ##### EXTRACT PATCHES
        cluster_map = []
        all_patches = []
        starts = []
        ends =[]
        shapes = []
        start = 0 
        n_patches_tot = []
        indices = []

        for (image, mask) in zip(x, masks):
            shapes.append(image.shape[0:2])
            patches, idx, n_patches = autoencoder_predict.extract_patches(image, mask, mask_threshold=0.9)  # Assuming this function extracts and reshapes patches for a single image
            #patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
            
            n_patches_cao = len(patches)
            all_patches.append(patches)
            starts.append(start)
            ends.append(start + n_patches_cao)
            n_patches_tot.append(n_patches)
            indices.append(idx)
            start += n_patches_cao

        # Stack filtered patches from all images
        patches = np.concatenate(all_patches, axis=0) / max_vals

        ##### ENCODE THE PATCHES AND PREDICT
        encoded_patches = encoder.predict(patches)
        encoded_patches_flat = encoded_patches.reshape(encoded_patches.shape[0], -1)

        print("loaded cluster")
        labels = cluster.predict(encoded_patches_flat)


        ##### PERFORM CHECK

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
            current_labels = np.ones((n_patches_tot[i]))*(global_max+1)
            current_labels[np.squeeze(indices[i].numpy())] = labels[starts[i]:ends[i]]
        
            label_map = np.reshape(current_labels, (reduced_height, reduced_width))
            plt.imshow
            binary_map = (label_map == desired_label)

            # Label connected components
            labeled_map, num_features = ndimage.label(binary_map)

            # Measure sizes of connected components
            region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

            # Check if any region exceeds the size threshold
            if any(region_sizes > size_threshold):

                if dates[i] not in selected_dates:
                    selected_dates.append(dates[i]) 
                    # fig, axs = plt.subplots(1,2)
                    # fig.suptitle("%s\nCAO found for threshold %s" %(dates[i], size_threshold))
                    # axs[0].imshow(x[i], cmap="gray")
                    # axs[0].invert_yaxis()
                    # axs[0].invert_xaxis()

                    # #axs[0].invert_xaxis()

                    # cb =axs[1].imshow(label_map, cmap="tab10", norm=norm)   
                    # axs[1].invert_yaxis()
                    # axs[1].invert_xaxis()


                    # plt.colorbar(cb)
                    # plt.savefig("/uio/hume/student-u37/fslippe/master_project/figures/CAO_found_at_%s" %(dates[i])) 

                    # plt.show()
        print(start_date, end_date)
        np.save("/uio/hume/student-u37/fslippe/data/date_hist_%s-%s_%s" %(start_date, end_date, size_threshold), np.array(selected_dates))
    #np.save("/uio/hume/student-u37/fslippe/data/date_hist_202104+2_%s" %(size_threshold), np.array(selected_dates))
