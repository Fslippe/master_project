from keras.layers import Input, Dense, Flatten, Reshape
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
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

from pyhdf.SD import SD, SDC
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap, cm


folder = "/home/filip/Documents/master_project/training_set/data/"
all_files = os.listdir(folder)
dataset = np.load(folder + all_files[0])
dataset_1 = np.load(folder + all_files[1])
dataset_2 = np.load(folder + all_files[2])
dataset_3 = np.load(folder + all_files[3])
dataset_4 = np.load(folder + all_files[4])
dataset_5 = np.load(folder + all_files[5])
dataset_6 = np.load(folder + all_files[6])
dataset_7 = np.load(folder + all_files[7])
dataset_7 = np.nan_to_num(dataset_7, -1)

data_list = [dataset, dataset_1, dataset_2,
             dataset_3, dataset_4, dataset_5, dataset_6]
min_shape = np.min([data.shape[0] for data in data_list]), np.min(
    [data.shape[1] for data in data_list])

preprocessed_data = []
for data in data_list:
    data = np.nan_to_num(data, nan=-1)
    preprocessed_data.append(data[:min_shape[0], :min_shape[1]])

# Concatenate all the datasets into one
X = np.concatenate(preprocessed_data, axis=0)

input_shape = X.shape[1:]


latent_dim = 64

# Check if the data is already 1D or 2D
if len(input_shape) == 1:
    flattened_input = Input(shape=input_shape)
else:
    inputs = Input(shape=input_shape)
    flattened_input = Flatten()(inputs)

# Encoder
encoded = Dense(512, activation='relu')(flattened_input)
encoded = Dense(latent_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)
decoded = Reshape(input_shape)(decoded)

autoencoder = Model(flattened_input, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

patch_size = (10, 10)  # or another appropriate size
X_patches = [extract_patches_2d(data, patch_size)
             for data in preprocessed_data]
X_patches = np.vstack(X_patches)

input_shape = patch_size

# ... [your autoencoder model definition]

# Train the autoencoder on the patches
autoencoder.fit(X_patches, X_patches, epochs=50, batch_size=256)

# Predict (reconstruct) the patches
decoded_patches = autoencoder.predict(X_patches)

# Now, if you wish to view the entire reconstructed image (not necessary for clustering, but might be informative):
decoded_img = reconstruct_from_patches_2d(
    decoded_patches, (len(preprocessed_data) * min_shape[0], min_shape[1]))

# To cluster, we'll first get the compressed representation of the patches:
# Assuming the encoding layer is 3 layers from the end
encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.layers[-3].output)
encoded_patches = encoder.predict(X_patches)

# Flatten encoded patches for clustering
encoded_patches_flattened = encoded_patches.reshape(
    encoded_patches.shape[0], -1)

# Cluster the encoded patches
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
    encoded_patches_flattened)
labels = kmeans.labels_

# Reshape the labels to the size of your image
label_patches = labels.reshape(-1, 1, 1).repeat(
    patch_size[0], axis=1).repeat(patch_size[1], axis=2)
labels_img = reconstruct_from_patches_2d(
    label_patches, (len(preprocessed_data) * min_shape[0], min_shape[1]))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

cmp = axes[0].imshow(dataset_2, cmap='gray')
plt.colorbar(cmp, ax=axes[0])
axes[0].set_title('Original Image')
axes[0].axis('off')

cmp = axes[1].imshow(labels_img, cmap='tab10')
plt.colorbar(cmp, ax=axes[1])
axes[1].set_title('Clustered Image')
axes[1].axis('off')

plt.show()
