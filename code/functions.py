from datetime import datetime, timedelta 
import numpy as np 
from scipy import ndimage
from autoencoder import * 

def convert_to_day_of_year(date_str):
    # Parse the date
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Convert to datetime object
    date_obj = datetime(year, month, day)

    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday

    # Return in the desired format
    return f"{year}{day_of_year:03d}"  # Using :03d to ensure it's a 3-digit number



def convert_to_standard_date(date_str):
    # Parse the date
    year = int(date_str[:4])
    day_of_year = int(date_str[4:])

    # Convert to datetime object
    date_obj = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)  # Using day_of_year - 1 because timedelta is 0-indexed

    # Return in the desired format
    return date_obj.strftime('%Y%m%d')

def generate_map_from_labels(labels, start, end, shape, idx, global_max, n_patches, patch_size):
    # Calculate the dimensions of the reduced resolution array
    height, width = shape
    reduced_height = height // patch_size
    reduced_width = width //patch_size

    # Generate map with empty land clusters 
    current_labels = np.ones((n_patches))*(global_max+1)
    current_labels[np.squeeze(idx.numpy())] = labels[start:end]
    cluster_map =  np.reshape(current_labels, (reduced_height, reduced_width))

    return cluster_map

def generate_map_from_patches(patches, start, end, shape, patch_size):
    # Compute the number of patches in each dimension
    num_patches_y = shape[0] // patch_size
    num_patches_x = shape[1] // patch_size
    
    # Calculate the reduced resolution based on number of patches and patch size
    reduced_height = num_patches_y * patch_size
    reduced_width = num_patches_x * patch_size

    # Create an empty map of the reduced resolution
    reconstructed_image = np.zeros((reduced_height, reduced_width))

    # Extract the patches corresponding to this image
    image_patches = patches[start:end]

    # Place each patch into the empty image
    patch_idx = 0
    for y in range(0, reduced_height, patch_size):
        for x in range(0, reduced_width, patch_size):
            # Check if we've used all our patches
            if patch_idx >= len(image_patches):
                break
            # Place the patch in the correct position
            reconstructed_image[y:y+patch_size, x:x+patch_size] = image_patches[patch_idx]
            patch_idx += 1

    return reconstructed_image

def generate_patches(x, masks, lon_lats, max_vals, autoencoder, strides = [None, None, None, None]):
    all_patches = []
    all_lon_patches = []
    all_lat_patches = []

    starts = []
    ends =[]
    shapes = []
    start = 0 
    n_patches_tot = [] 
    indices = []


    #encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
    #normalized_patches = np.concatenate([autoencoder.extract_patches(n_d) for n_d in normalized_data], axis=0)

    i=0
    tot = len(x)
    for (image, mask, lon_lat) in zip(x, masks, lon_lats):
        print("%s/%s" %(i, tot))
        shapes.append(image.shape[0:2])
        patches, idx, n_patches, lon, lat = autoencoder.extract_patches(image,
                                                                            mask,
                                                                            mask_threshold=0.9,
                                                                            lon_lat=lon_lat,
                                                                            extract_lon_lat=True,
                                                                            strides=strides)  # Assuming this function extracts and reshapes patches for a single image
        #patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
        #n_patches = len(patches)
        

        all_patches.append(patches)
        all_lon_patches.append(lon)
        all_lat_patches.append(lat)
    

        starts.append(start)
        ends.append(start + len(patches))
        n_patches_tot.append(n_patches)
        indices.append(idx)
        start += len(patches)
        i+=1
    # Stack filtered patches from all images
    patches = np.concatenate(all_patches, axis=0) / max_vals

    return patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices


def get_patches_of_img_cao(labels, patches, starts, ends, shapes, indices, global_max, n_patches_tot, desired_label, size_threshold, n,  patch_size):
    """
    Find pictures with regions of patches of a desired label of sizes higher than given threshold 
    """
    patches_w = []

    for i in range(n):
        label_map = generate_map_from_labels(labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)
        
        binary_map = (label_map == desired_label)
        
        # Label connected components
        labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))
        # Iterate through each region and check if its size exceeds the threshold

        for region_idx, region_size in enumerate(region_sizes):
            if region_size > size_threshold:
                patches_w.append(patches[starts[i]:ends[i]])
    patches_w = np.concatenate(patches_w)
    return patches_w