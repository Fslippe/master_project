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
from keras.optimizers import Adam
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pyhdf.SD import SD, SDC
import matplotlib as mpl
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

from mpl_toolkits.basemap import Basemap, cm
import os
tf.config.threading.set_inter_op_parallelism_threads(1)

class SobelFilterLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        sobel_x = tf.image.sobel_edges(inputs)[:, :, :, :, 0]  # Sobel filter along x-axis
        sobel_y = tf.image.sobel_edges(inputs)[:, :, :, :, 1]  # Sobel filter along y-axis
        sobel = tf.sqrt(sobel_x**2 + sobel_y**2)
        return sobel


class SimpleAutoencoder:
    def __init__(self, n_vars, patch_size, patch_size_2=None):
        self.n_vars = n_vars
        self.patch_size = patch_size
        if patch_size_2 == None:
            self.patch_size_2 = self.patch_size
        else:
            self.patch_size_2 = patch_size_2
    
    
    def extract_patches(self, image):     
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        else:
            assert image.ndim == 4, "Image has wrong shape. Epected dimension 4, has dimension %s" %image.ndim 
        sizes = [1, self.patch_size, self.patch_size_2, 1]
        strides = [1, self.patch_size, self.patch_size_2, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'  # 'VALID' ensures no padding and might discard right/bottom parts of the image if not fitting exactly
        patches = tf.image.extract_patches(images=image,
                                        sizes=sizes,
                                        strides=strides,
                                        rates=rates,
                                        padding=padding)

        # Reshaping the tensor for easier indexing of patches
        # patches_reshaped =  tf.reshape(patches, (-1, self.patch_size, self.patch_size_2, image.shape[-1]))
        # Check for NaN values in patches and create a mask
        # mask = tf.math.logical_not(tf.math.is_nan(patches_reshaped))
        
        # Check if all values in the patch are not NaN
        # valid_mask = tf.reduce_all(mask, axis=(1,2,3))

        # Use boolean mask to filter patches
        # valid_patches = tf.boolean_mask(patches_reshaped, valid_mask)
        
        #return valid_patches
        return tf.reshape(patches, (-1, self.patch_size, self.patch_size_2, image.shape[-1]))
    def encode(self):
        #encoder_input = keras.Input(shape=(patch_size, patch_size, 1))
        self.encoder_input = keras.Input(shape=(self.patch_size, self.patch_size_2, self.n_vars)) 
        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.encoder_input)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        self.encoded = keras.layers.MaxPooling2D((2, 2))(x)
        self.encoder = keras.Model(self.encoder_input, self.encoded)

    def decode(self):
        decoder_input = keras.Input(shape=(self.encoded.shape[1], self.encoded.shape[2], self.encoded.shape[3]))
        x = keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decoder_input)
        x = keras.layers.UpSampling2D((2, 2))(x)
        #x = SobelFilterLayer()(x)
        x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

        x = keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        # Final layer to get back to original depth
        decoded = keras.layers.Conv2DTranspose(self.n_vars, (3, 3), activation='sigmoid', padding='same')(x)
        
        self.decoder = keras.Model(decoder_input, decoded)


    def fit(self, datasets, epochs, batch_size, loss="mse", threshold = 0.1, optimizer = "adam", predict_self=False):
        normalized_datasets = self.normalize(datasets)

        normalized_datasets = np.nan_to_num(normalized_datasets, nan=-1)

        all_patches = []

        for image in normalized_datasets:
            patches = self.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
            
            # Filter the patches for the current image
            mask = np.mean(patches, axis=(1,2,3)) > threshold
            filtered_patches = patches[mask]

            all_patches.append(filtered_patches)

        # Stack filtered patches from all images
        self.patches = np.concatenate(all_patches, axis=0)
        
        print("Patches shape: ", self.patches.shape)
        #self.patches = self.extract_patches(normalized_datasets)
        print("ENCODE")
        self.encode()
        print("DECODE")

        self.decode()
        print("LOSS")
        if loss == "mse":
            loss_func = "mse"
        elif loss=="combined":
            loss_func = self.combined_loss
        
        print("MODEL")

        self.autoencoder = keras.Model(self.encoder_input, self.decoder(self.encoded))
         
        print("COMPILE")

        self.autoencoder.compile(optimizer=optimizer, loss=loss_func)  # Using combined loss
        print("FIT")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.autoencoder.fit(self.patches, self.patches, epochs=epochs, batch_size=batch_size)
        print("FIT")

        if predict_self:
            self.predict()


    def normalize(self, data):
        normalized_data = []
        for i in range(len(data)):
            normalized_data.append((data[i] - np.nanmin(data[i], axis=(0,1), keepdims=True)) / (np.nanmax(data[i], axis=(0,1), keepdims=True) - np.nanmin(data[i], axis=(0,1), keepdims=True)))
        #normalized_data = (data - np.nanmin(data, axis=(1,2), keepdims=True)) / (np.nanmax(data, axis=(1,2), keepdims=True) - np.nanmin(data, axis=(1,2), keepdims=True))
        return normalized_data
    
    def predict(self, datasets):
        if datasets.shape[1] != self.patch_size and datasets.shape[2] != self.patch_size_2:
            patches = self.extract_patches(datasets)
        else:
            patches = datasets 

        return self.autoencoder.predict(patches)
    
        
        
    def sobel_loss(self, y_true, y_pred):
        sobel_true = tf.image.sobel_edges(y_true)
        sobel_pred = tf.image.sobel_edges(y_pred)

        # Compute L2 (MSE) loss on Sobel-filtered images
        sobel_loss = tf.reduce_mean(tf.square(sobel_true - sobel_pred))

        return sobel_loss
    
    def combined_loss(self, y_true, y_pred, alpha=0.5):
        # MSE or L2 loss
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Sobel loss
        sbl_loss = self.sobel_loss(y_true, y_pred)
        
        return mse + alpha * sbl_loss

    def kmeans(self, data, n_clusters=10, random_state=None):
        patches = self.extract_patches(data)
        encoded_patches = self.encoder.predict(patches)

        # Flatten the encoded patches for clustering
        self.encoded_patches_flat = encoded_patches.reshape(encoded_patches.shape[0], -1)

        # KMeans clustering
        if random_state != None:
            kmeans = KMeans(n_clusters, random_state=random_state).fit(self.encoded_patches_flat)
        else:
            kmeans = KMeans(n_clusters).fit(self.encoded_patches_flat)

        labels = kmeans.labels_
        # Assuming your original data shape is (height, width)
        height, width = data.shape[1:3]

        # Calculate the dimensions of the reduced resolution array
        reduced_height = height // self.patch_size
        reduced_width = width // self.patch_size_2

        cluster_map = np.reshape(labels, (reduced_height, reduced_width))
        return cluster_map
    
    def clustering_agglomerative(self, n_clusters=2, affinity='euclidean', linkage='ward', data_shape=(2030, 1354)):
        """
        Clusters the encoded patches using Agglomerative Hierarchical Clustering.
        
        Parameters:
        - n_clusters: The number of clusters to find.
        - affinity: Metric used to compute linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
        - linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation.
        
        Returns:
        - A 2D array of cluster labels.
        """
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        labels = agglomerative.fit_predict(self.encoded_patches_flat)

        # Assuming your original data shape is (height, width)
        height, width = data_shape

        # Calculate the dimensions of the reduced resolution array
        reduced_height = height // self.patch_size
        reduced_width = width // self.patch_size_2

        cluster_map = np.reshape(labels, (reduced_height, reduced_width))
        return cluster_map
    
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
print("HEIG")

folder = "/nird/projects/NS9600K/data/modis/cao/"
file_name = "/nird/projects/NS9600K/data/modis/cao/MOD021KM.A2021080.1300.061.2021081011315.hdf"
hdf = SD(file_name, SDC.READ)
bands = [6]#, 7, 20]

list1 = [int(num_str) for num_str in hdf.select("EV_250_Aggr1km_RefSB").attributes()["band_names"].split(",")]
list2 = [int(num_str) for num_str in hdf.select("EV_500_Aggr1km_RefSB").attributes()["band_names"].split(",")]
list3 = [int(num_str) for num_str in hdf.select("EV_1KM_RefSB").attributes()["band_names"].split(",") if num_str.isdigit()]
list4 = [int(num_str) for num_str in hdf.select("EV_1KM_Emissive").attributes()["band_names"].split(",")]

file_layers = np.empty(36, dtype=object)
for i, (band) in enumerate(list1):
    file_layers[band-1] = {"EV_250_Aggr1km_RefSB": i}
for i, (band) in enumerate(list2):
    file_layers[band-1] = {"EV_500_Aggr1km_RefSB": i}    
for i, (band) in enumerate(list3):
    file_layers[band-1] = {"EV_1KM_RefSB": i}
for i, (band) in enumerate(list4):
    file_layers[band-1] = {"EV_1KM_Emissive": i}


all_files = os.listdir(folder)[0:1]

X = np.empty((len(all_files), 2030, 1354, len(bands)))

for i, (file) in enumerate(all_files):
    hdf = SD(folder + file, SDC.READ)
    for j, (band) in enumerate(bands):
        key = list(file_layers[band-1].keys())[0]
        idx = list(file_layers[band-1].values())[0]

        attrs = hdf.select(key).attributes()
        data = hdf.select(key)[:][idx]
        is_nan = (np.where(data == attrs["_FillValue"]))
        data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]

        if not len(is_nan[0]) == 0:
            data = data[is_nan[0][-1]+1:, :] if is_nan[1][-1] == 1353 else data[:, is_nan[1][-1]+1:]

        X[i, :, :, j] = data


X = X[:,::4, ::4, :]
autoencoder = SimpleAutoencoder(1, 64, 64)
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
autoencoder.fit(X, epochs=1, batch_size=8, optimizer="adam", threshold=0.09,loss="combined")
# print(X[0].shape)
# #autoencoder = simple_autoencoder([data_01], patch_size)
# autoencoder = simple_autoencoder(1, (2040, 1354), patch_size)    
# autoencoder.fit(X, epochs=5, batch_size=256)