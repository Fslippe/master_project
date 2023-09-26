# %%
import os
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
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.model_selection import train_test_split




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

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

#bands = [6, 7, 20, 28, 28, 31]


# %%
bands = [6,20,29]


# %%


# %%
# x2 = [xi for xi in  extract_1km_data(folder, bands=[6], start_date=start_converted, end_date=end_converted) if xi.shape[0] > 64]
# x3 = [xi for xi in  extract_1km_data(folder, bands=[1], start_date=start_converted, end_date=end_converted) if xi.shape[0] > 64]
# x4 = [xi for xi in  extract_1km_data(folder, bands=[7], start_date=start_converted, end_date=end_converted) if xi.shape[0] > 64]
# x5 = [xi for xi in  extract_1km_data(folder, bands=[20], start_date=start_converted, end_date=end_converted) if xi.shape[0] > 64]


# %%
# fig, axs = plt.subplots(1,5, dpi=300)
# i = 15
# cb = axs[4].imshow(x[i][:,:,0], cmap="gray")
# plt.colorbar(cb)
# axs[1].imshow(x2[i], cmap="gray")
# axs[0].imshow(x3[i], cmap="gray")
# axs[2].imshow(x4[i], cmap="gray")
# axs[3].imshow(x5[i], cmap="gray")



# %%
import autoencoder
import importlib
importlib.reload(autoencoder)
from autoencoder import SobelFilterLayer, SimpleAutoencoder
patch_size = 64

autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size)
#x = autoencoder.normalize(x)
#optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=1e-4), loss_scale='dynamic')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model = autoencoder.model(optimizer=optimizer, loss="combined")


# %%

#val_data = np.load("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_val_patches")
patches = np.load("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_train_patches.npy")[::2]

# %%
# Splitting the data

# %%
print(patches.shape)

# %%
# %%
### SAVE TRAIN TEST PATCHES 
# patches = np.load("/scratch/fslippe/modis/MOD02/training_data/normalized_trainingpatches_bands6,20,29_winter20_21.npy")[::4]
# val_data  = np.load("/scratch/fslippe/modis/MOD02/test_data/normalized_testpatches_bands6,20,29_winter20_21.npy")[::4]
# patches.shape

# print(np.mean(patches, axis=(0,1,2)))

# %%
### STANDARD FIT
gc.collect()
model.fit(patches, patches, epochs=200, batch_size=32)


# %%
# #### FIT  USING TF.data API pipelining
# patches = tf.data.Dataset.from_tensor_slices(patches)
# patches = patches.batch(10)
# def data_generator():
#     """Generator to yield data from files."""
#     #random.shuffle(file_list)  # Shuffle files at the beginning of each epoch
#     for patches_chunk in patches:
#         #data = np.load(file_name)
#         for item in patches_chunk:
#             yield item

# # Define your dataset
# dataset = tf.data.Dataset.from_generator(data_generator,
#                                          output_signature=(tf.TensorSpec(shape=(...), dtype=tf.float32)))  # Fill in the shape and type
# dataset = dataset.shuffle(buffer_size=10000)  # Shuffle data
# dataset = dataset.batch(32)  # Batch data
# dataset = dataset.repeat()  # Repeat dataset indefinitely
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch data
# steps_per_epoch = len(total_data) // batch_size
# model.fit(dataset, epochs=100, steps_per_epoch=steps_per_epoch, validation_data=(val_data, val_data))

# %%
#model.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_autoencoder")
autoencoder.encoder.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
autoencoder.decoder.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_decoder")


# %%
# np.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_val_patches", val_data)
# np.save("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_train_patches", patches)



