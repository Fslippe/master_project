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
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#bands = [6, 7, 20, 28, 28, 31]
bands=[29]
#bands=[1]
folder = "/scratch/fslippe/modis/MOD02/daytime_1km/ /scratch/fslippe/modis/MOD02/boundary_1km/ /scratch/fslippe/modis/MOD02/night_1km/"





# %%
import autoencoder
import importlib
importlib.reload(autoencoder)
from autoencoder import SobelFilterLayer, SimpleAutoencoder
patch_size = 64
#normalized_patches = np.concatenate([autoencoder.extract_patches(n_d) for n_d in normalized_data], axis=0)

# %%
from autoencoder import SobelFilterLayer, SimpleAutoencoder
patch_size = 64
print(len(bands))
autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)

#encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_scheduler_band(29)_filter_encoder")
max_vals = np.load("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_band(29)_max_vals.npy")


# %%
# %%
#cluster = MiniBatchKMeans(11, batch_size=32, random_state=42).fit(encoded_patches_flat)
#joblib.dump(cluster, '/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_scheduler_band(29)_filter_cluster.pkl')

# %%
### EXTRACT CAO AND NOn CAO CASES
import importlib
import extract_training_data
importlib.reload(extract_training_data)
from extract_training_data import *
# start = "20230303"
# end = "20230306"
start = "20201201"
end = "20210401"
#start = "20210701"
#end = "20210702"
start_converted = convert_to_day_of_year(start)
end_converted = convert_to_day_of_year(end)
#x_cao, dates_cao, masks_cao = extract_1km_data("/scratch/fslippe/modis/MOD02/july_2021/", bands=bands, start_date=start_converted, end_date=end_converted)

# x_cao, dates_cao, masks_cao, lon_lats = extract_1km_data("/scratch/fslippe/modis/MOD02/cao_test_data/",
#                                                          bands=bands,
#                                                          start_date=start_converted,
#                                                          end_date=end_converted,
#                                                          return_lon_lat=True)
x_cao, dates_cao, masks_cao, lon_lats = extract_1km_data(folder,
                                                         bands=bands,
                                                         start_date=start_converted,
                                                         end_date=end_converted,
                                                         return_lon_lat=True)
x_cao, dates_cao, masks_cao, lon_lats = zip(*[(xi, date, mask, lon_lat) for xi, date, mask, lon_lat in zip(x_cao, dates_cao, masks_cao, lon_lats) if (xi.shape[0] > 64) and (xi.shape[1] > 64)])

x_cao = list(x_cao)
dates_cao = list(dates_cao)


# %%
#### EXTRACTING AND ENCODING PATCHES + SAVING indexes of belonging files
import functions
importlib.reload(functions)
from functions import *
cluster_map_cao = []
patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices = generate_patches(x_cao,
                                                                                                                                       masks_cao,
                                                                                                                                       lon_lats,
                                                                                                                                       max_vals,
                                                                                                                                       autoencoder_predict)

# %%
n_patches = len(patches_cao)
n_patches

# %%
encoded_patches_cao_1 = encoder.predict(patches_cao[:n_patches // 2])
encoded_patches_cao_2 = encoder.predict(patches_cao[n_patches // 2:])

encoded_patches_cao = np.concatenate([encoded_patches_cao_1, encoded_patches_cao_2])
encoded_patches_flat_cao = encoded_patches_cao.reshape(encoded_patches_cao.shape[0], -1)

gc.collect()

# %%
cluster = joblib.load('/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_scheduler_band(29)_filter_cluster_lab2.pkl')

labels = cluster.predict(encoded_patches_flat_cao)


# %%

global_max = np.max([np.max(cm) for cm in cluster.labels_])+2


# %%
patches_cao.shape
desired_label = 2
size_threshold = 15
import functions
importlib.reload(functions)
from functions import *

patches_w_cao = get_patches_of_img_cao(labels, patches_cao, starts_cao, ends_cao, shapes_cao, indices, global_max, n_patches_tot_cao, desired_label, size_threshold, len(dates_cao),  patch_size)


train, test = train_test_split(patches_w_cao, test_size=0.15)


# %%


# %%
# TRY TRAINING ON PATCHES EXTRACTED
patch_size = 64
bands = [1]  # You might need to specify the bands here
autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size)

# Set up your optimizer and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model = autoencoder.model(optimizer=optimizer, loss="combined")

# Train the model on your dataset
batch_size = 32 
from read_tf import scheduler
lr_schedule = LearningRateScheduler(scheduler, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

history = model.fit(train, train, validation_data=(test, test), epochs=500, callbacks=[early_stopping, lr_schedule])

model.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_scheduler_band(29)_filter_secondary_autoencoder")
autoencoder.encoder.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_scheduler_band(29)_filter_secondary_encoder")
autoencoder.decoder.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_dnb_landmask_150k_scheduler_band(29)_filter_secondary_decoder")








