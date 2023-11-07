import tensorflow as tf
import numpy as np 
import keras
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from concurrent.futures import ProcessPoolExecutor

#tf.config.threading.set_inter_op_parallelism_threads(128)


class SobelFilterLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        sobel_x = tf.image.sobel_edges(inputs)[:, :, :, :, 0]  # Sobel filter along x-axis
        sobel_y = tf.image.sobel_edges(inputs)[:, :, :, :, 1]  # Sobel filter along y-axis
        sobel = tf.sqrt(sobel_x**2 + sobel_y**2)
        return sobel


class SimpleAutoencoder:
    def __init__(self, n_vars, patch_size, patch_size_2=None, filters = [16, 32, 64, 128]):
        self.n_vars = n_vars
        self.patch_size = patch_size
        if patch_size_2 == None:
            self.patch_size_2 = self.patch_size
        else:
            self.patch_size_2 = patch_size_2
        self.filters = filters
        
    
    def valid_percentage(self, mask_patch):
        return tf.reduce_mean(tf.cast(mask_patch, tf.float32))

    def within_roi(self, lon_patch, lat_patch, lon_min, lon_max, lat_min, lat_max):
        # Get the mean latitude and longitude for the patch
        mean_lon = tf.reduce_mean(lon_patch)
        mean_lat = tf.reduce_mean(lat_patch)
        
        lon_valid = tf.logical_and(mean_lon <= lon_max, mean_lon >= lon_min)
        lat_valid = tf.logical_and(mean_lat <= lat_max, mean_lat >= lat_min)
        
        return tf.logical_and(lon_valid, lat_valid)

    def filter_patches(self, image_patches, mask_patches, lon_patches, lat_patches, threshold=0.9, 
                    lon_lat_min_max=[-35,35,60,82]):
        """ FILTERING BASED ON BOTH LAND PERCENTAGE AND MEAN LON LAT OF EACH PATCH INSIDE THRESHOLD"""
        
        percentages = tf.map_fn(self.valid_percentage, np.float32(mask_patches))
        roi_mask = tf.map_fn(lambda x: self.within_roi(x[0], x[1], lon_lat_min_max[0], lon_lat_min_max[1], lon_lat_min_max[2], lon_lat_min_max[3]), 
                            (lon_patches, lat_patches), dtype=tf.bool)

        mask = tf.logical_and(percentages >= threshold, roi_mask)
        
        filtered_image_patches = tf.boolean_mask(image_patches, mask)
        valid_indices = tf.where(mask)

        return filtered_image_patches, valid_indices


    
    def extract_patches(self, image, mask=None, lon_lat=None, mask_threshold=None, extract_lon_lat=False, strides=[None, None, None, None], lon_lat_min_max=[-35, 35, 60, 82]):
        # Expand dimensions if the image is 3D
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 2:
            # Assuming single-channel (grayscale) image, expand both batch and channel dimensions
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)

        with tf.device('/CPU:0'):   
            # Expand dimensions if the image is 3D

            sizes = [1, self.patch_size, self.patch_size_2, 1]
            if strides[0] == None:
                strides = [1, self.patch_size, self.patch_size_2, 1]
            
            rates = [1, 1, 1, 1]
            padding = 'VALID'

            patches = tf.image.extract_patches(images=image,
                                            sizes=sizes,
                                            strides=strides,
                                            rates=rates,
                                            padding=padding)
            
            if mask_threshold != None:
                if mask.ndim == 3:
                    mask = np.expand_dims(mask, axis=0)
                elif mask.ndim == 2:
                    # Assuming single-channel (grayscale) mask, expand both batch and channel dimensions
                    mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=-1)
                
                mask_patches = tf.image.extract_patches(images=mask,
                                sizes=sizes,
                                strides=strides,
                                rates=rates,
                                padding=padding)
            
                mask_patches = tf.reshape(mask_patches, (-1, self.patch_size, self.patch_size_2, 1))

            patches = tf.reshape(patches, (-1, self.patch_size, self.patch_size_2, image.shape[-1]))
            if extract_lon_lat:
                if lon_lat.ndim == 3:
                    lon_lat = np.expand_dims(lon_lat, axis=-1)
                elif lon_lat.ndim == 2:
                    # Assuming single-channel (grayscale) lon_lat, expand both batch and channel dimensions
                    lon_lat = np.expand_dims(np.expand_dims(lon_lat, axis=0), axis=-1)
                    
                lon = tf.image.extract_patches(images=np.expand_dims(lon_lat[0], axis=0),
                                            sizes=sizes,
                                            strides=strides,
                                            rates=rates,
                                            padding=padding)
                
                lat = tf.image.extract_patches(images=np.expand_dims(lon_lat[1], axis=0),
                                            sizes=sizes,
                                            strides=strides,
                                            rates=rates,
                                            padding=padding)
                
                lon = tf.reshape(lon, (-1, self.patch_size, self.patch_size_2))
                lat = tf.reshape(lat, (-1, self.patch_size, self.patch_size_2))
                
            if mask_threshold != None and extract_lon_lat:
                n_patches = len(patches)
                patches, idx = self.filter_patches(patches, mask_patches, lon, lat, threshold=mask_threshold, lon_lat_min_max=lon_lat_min_max)
                if extract_lon_lat:
                    idx_tf = tf.convert_to_tensor(np.squeeze(idx.numpy()), dtype=tf.int32)
                    lon = tf.gather(lon, idx_tf)
                    lat = tf.gather(lat, idx_tf)

                    return patches, idx, n_patches, lon, lat
                else:
                    return patches, idx, n_patches
            else:
                return patches

    
    def residual_block(self, x, filters):
        """
        Build a residual block with the specified number of filters.
        """
        res = keras.layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(x)
        res = keras.layers.LeakyReLU(alpha=0.3)(res)
        res = keras.layers.Conv2D(filters, (3, 3), padding='same')(res)

        x = keras.layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(x)
        x = keras.layers.add([x, res])
        x = keras.layers.BatchNormalization()(x)
        return keras.layers.LeakyReLU(alpha=0.3)(x)

    
    def encode(self):
        self.encoder_input = keras.Input(shape=(self.patch_size, self.patch_size_2, self.n_vars)) 

        x = self.residual_block(self.encoder_input, self.filters[0])
        x = self.residual_block(x, self.filters[1])
        x = self.residual_block(x, self.filters[2])
        self.encoded = self.residual_block(x, self.filters[3])
        self.encoder = keras.Model(self.encoder_input, self.encoded)

    def decode(self):
        decoder_input = keras.Input(shape=(self.encoded.shape[1], self.encoded.shape[2], self.encoded.shape[3]))

        x = keras.layers.Conv2DTranspose(self.filters[3], (3, 3), padding='same')(decoder_input)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        x = keras.layers.Conv2DTranspose(self.filters[2], (3, 3), padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        x = keras.layers.Conv2DTranspose(self.filters[1], (3, 3), padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

        x = keras.layers.Conv2DTranspose(self.filters[0], (3, 3), padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        x = keras.layers.Conv2DTranspose(self.n_vars, (3, 3), padding='same')(x)
        decoded = keras.layers.LeakyReLU(alpha=0.3)(x)

        
        self.decoder = keras.Model(decoder_input, decoded)

    
    def model(self, loss="mse", threshold = 0.1, optimizer = "adam"):
        print("Input should already be normalized. Call self.normalize to normalize list of data")
        self.encode()
        self.decode()
        if loss == "mse":
            loss_func = "mse"
        elif loss=="combined":
            loss_func = self.combined_loss
        model = keras.Model(self.encoder_input, self.decoder(self.encoded))
        model.compile(optimizer=optimizer, loss=loss_func)  # Using combined loss

        return model

    def fit(self, normalized_datasets, epochs, batch_size, loss="mse", threshold = 0.1, optimizer = "adam", predict_self=False):
        print("Input should already be normalized. Call self.normalize to normalize list of data")
        all_patches = []

        print("Extracting patches...")
        tot_pics = len(normalized_datasets)
        for i, image in enumerate(normalized_datasets):
            print("Extracting image", i, "of", tot_pics)
            patches = self.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
            
            # Filter the patches for the current image
            mask = np.mean(patches, axis=(1,2,3)) > threshold
            filtered_patches = patches[mask]

            all_patches.append(filtered_patches)

        # Stack filtered patches from all images
        self.patches = np.concatenate(all_patches, axis=0)
        
        print("Patches shape: ", self.patches.shape)
        #self.patches = self.extract_patches(normalized_datasets)
        self.encode()
        self.decode()
        if loss == "mse":
            loss_func = "mse"
        elif loss=="combined":
            loss_func = self.combined_loss
        self.autoencoder = keras.Model(self.encoder_input, self.decoder(self.encoded))
         
        
        self.autoencoder.compile(optimizer=optimizer, loss=loss_func)  # Using combined loss
        self.autoencoder.fit(self.patches, self.patches, epochs=epochs, batch_size=batch_size)

        if predict_self:
            self.predict()

    def normalize(self, data):
        # Calculate global min and max
        global_min = float("inf")
        global_max = float("-inf")
        for item in data:
            current_min = np.min(item)
            current_max = np.max(item)
            
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

        # Normalize data
        normalized_data = [(item - global_min) / (global_max - global_min) for item in data]

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
    

    def clustering(self, datasets, n_clusters=10, encoder=None, random_state=None, normalize_max_val=[None], method="kmeans", batch_size=100, predict=True):
        cluster_map = []
        all_patches = []
        starts = []
        ends =[]
        shapes = []
        start = 0 

        for image in datasets:
            shapes.append(image.shape[0:2])
            patches = self.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
            all_patches.append(patches)
            starts.append(start)
            ends.append(start + len(patches))
            start += len(patches)


        
        # Stack filtered patches from all images
        patches = np.concatenate(all_patches, axis=0)
        print(patches.shape)
        if normalize_max_val[0] != None:
            patches = patches  / np.array(normalize_max_val)
            print("normalized patches")
        if encoder == None:
            encoded_patches = self.encoder.predict(patches)
        else:
            encoded_patches = encoder.predict(patches)
        print(encoded_patches.shape)
        self.encoded_patches_flat = encoded_patches.reshape(encoded_patches.shape[0], -1)
        
        # KMeans clustering
        if method=="minibatchkmeans":
            if random_state != None:
                cluster = MiniBatchKMeans(n_clusters, batch_size=batch_size, random_state=random_state).fit(self.encoded_patches_flat)
            else:
                cluster = MiniBatchKMeans(n_clusters, batch_size=batch_size).fit(self.encoded_patches_flat)
        elif method == "kmeans":
            if random_state != None:
                cluster = KMeans(n_clusters, random_state=random_state).fit(self.encoded_patches_flat)
            else:
                cluster = KMeans(n_clusters).fit(self.encoded_patches_flat)
        elif method == "agglomerative":
            if random_state != None:
                cluster = AgglomerativeClustering(n_clusters, random_state=random_state).fit(self.encoded_patches_flat)
            else:
                cluster = AgglomerativeClustering(n_clusters).fit(self.encoded_patches_flat)
        if predict:
            labels = cluster.labels_

            # Assuming your original data shape is (height, width)
            for i in range(len(datasets)):
                height, width = shapes[i]

                # Calculate the dimensions of the reduced resolution array
                reduced_height = height // self.patch_size
                reduced_width = width // self.patch_size_2
                cluster_map.append(np.reshape(labels[starts[i]:ends[i]], (reduced_height, reduced_width)))

            return cluster_map
        else:
            return cluster