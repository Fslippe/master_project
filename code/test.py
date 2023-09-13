from keras.layers import Input, Dense, Flatten, Reshape
from sklearn.feature_extraction import image as sk_image
from keras.models import Model
import os
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
#tf.config.threading.set_inter_op_parallelism_threads(128)

from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

from autoencoder import SobelFilterLayer, SimpleAutoencoder

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

bands = [6, 7, 20, 28, 28, 31]

loaded = np.load('/uio/hume/student-u37/fslippe/data/training_data/training_data_20210421.npz')
X = [loaded[key] for key in loaded][:4]
print(len(X))


X = [arr for arr in X if arr.shape[0] >= 64]

patch_size = 64
autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size)
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


autoencoder.fit(X, epochs=2, batch_size=64, optimizer="adam", threshold=0.09,loss="combined")
# print(X[0].shape)
X_test = autoencoder.normalize(X[:4])
print(np.max(X[0]))
print(np.max(X_test[0]))

cluster_map = autoencoder.kmeans(X_test, n_clusters=8)
print(len(cluster_map))
for i in range(3):
    fix, axs= plt.subplots(1,2, figsize=[10,8])

    cb = axs[0].imshow(cluster_map[i], cmap="tab10")
    plt.colorbar(cb)
    plt.tight_layout()
    cb = axs[1].imshow(X[i][:,:,0])
    plt.colorbar(cb)
    plt.tight_layout()

plt.show()