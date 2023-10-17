from datetime import datetime, timedelta 
import numpy as np 
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


def generate_patches(x, masks, lon_lats, max_vals, autoencoder):
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
                                                                            extract_lon_lat=True)  # Assuming this function extracts and reshapes patches for a single image
        #patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
        n_patches = len(patches)
        all_patches.append(patches)
        all_lon_patches.append(lon)
        all_lat_patches.append(lat)


        starts.append(start)
        ends.append(start + n_patches)
        n_patches_tot.append(n_patches)
        indices.append(idx)
        start += n_patches
        i+=1
    # Stack filtered patches from all images
    patches = np.concatenate(all_patches, axis=0) / max_vals

    return patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices