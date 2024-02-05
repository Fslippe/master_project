from pyhdf.SD import SD, SDC
import numpy as np 
import os 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import gc
from functions import *
import multiprocessing
from scipy.ndimage import zoom
from create_water_mask import * 
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, generic_filter
import socket

hostname = socket.gethostname()
if "nird" in hostname:
    data_loc = "/nird/projects/NS9600K/fslippe/data/"
if "mimi" in hostname:
    data_loc = "/uio/hume/student-u37/fslippe/data/"


total_cores = multiprocessing.cpu_count()
print("total cores:", total_cores)

def extract_1km_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/",
                     bands = [6,7,20,28,28,31],
                     ds_water_mask=None,
                     save=None,
                     start_date=None,
                     end_date=None,
                     date_list=None,
                     return_lon_lat=False,
                     workers=None,
                     max_zenith=50,
                     combine_pics=True,
                     data_loc="/uio/hume/student-u37/fslippe/data/",
                     data_type="hdf"):
    
    date_list = np.unique(np.array(date_list))
    if ds_water_mask==None:
        ds_water_mask = xr.open_dataset("%sland_sea_ice_mask/sea_land_mask.nc" %data_loc)
    all_files = []

    
    folders = folder.split(" ")
    print(folders)

    for f in folders:
        all_files.extend([os.path.join(f, file) for file in os.listdir(f) if file.endswith(data_type)])

    if data_type == "hdf":
        hdf = SD(all_files[0], SDC.READ)

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
    elif data_type == "mod06":
        hdf = SD(all_files[0], SDC.READ)

        file_layers[0] = {"EV_250_Aggr1km_RefSB": 0}
        for i, (band) in enumerate(list2):
            file_layers[band-1] = {"EV_500_Aggr1km_RefSB": i}    
        for i, (band) in enumerate(list3):
            file_layers[band-1] = {"EV_1KM_RefSB": i}
        for i, (band) in enumerate(list4):
            file_layers[band-1] = {"EV_1KM_Emissive": i}
    elif data_type == "npy":
        file_layers = None

    file_groups = defaultdict(list)
    mod_mins = defaultdict(list)


    # Loop through all files and group them by date
    for file in all_files:
        # Extract date from the filename (assuming the pattern is consistent)
        date = file.split('.')[1][1:]  # This will give e.g., '2021120' for 'MOD02QKM.A2021120'
        file_groups[date].append(file)
        mod_mins[date].append(int(file.split(".")[2]))


    sorted_keys = sorted(file_groups.keys(), key=int)  # Convert keys to integers for sorting

    # Extract the keys between start and end dates
    
    if start_date == None and end_date == None:
        print(sorted_keys)
        selected_keys = [key for key in date_list if key in sorted_keys]
        print(selected_keys)
    else:
        selected_keys = [key for key in sorted_keys if int(start_date) <= int(key) <= int(end_date)]
    #all_files = os.listdir(folder)[16:18]

    #X = np.empty((len(all_files), 2030, 1354, len(bands)))
    
    ds_all = []
    ##### FOLLOWING CODE IS USED IF PARALLELIZING OVER EACH DATE
    # if save == None:
    #     for key in list(selected_keys):
    #         file_group = file_groups[key]
    #         print("Date:", convert_to_standard_date(key))
    #         with ProcessPoolExecutor(max_workers=len(file_group)) as executor:
    #             X = list(tqdm(executor.map(append_data, [folder]*len(file_group), file_group, [file_layers]*len(file_group), [bands]*len(file_group), [min_mean]*len(file_group), [normalize]*len(file_group)), total=len(file_group)))
    #         ds_all.extend([xi for xi in X if xi.ndim>1])
    #     return ds_all
    

    water_mask_ravel = ds_water_mask["sea_ice_region_surface_mask"].values.ravel()
    coords_lowres = np.vstack((ds_water_mask.latitude.values.ravel(), ds_water_mask.longitude.values.ravel())).T
    tree = cKDTree(coords_lowres)
    
    ##### FOLLOWING CODE IS USED IF PARALLELIZING all files
    if workers == None:
        if len(selected_keys) < 10:
            workers = len(selected_keys)
        else:
            workers = 128
    if workers == 1:        
        if save is None:
            results = []
            for key in tqdm(selected_keys, total=len(selected_keys)):
                result = process_key(
                    key,
                    file_groups,
                    file_layers,
                    bands,
                    water_mask_ravel,
                    tree,
                    mod_mins,
                    return_lon_lat,
                    max_zenith
                )
                results.append(result)
            
            if return_lon_lat:
                ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]
                lon_lats = [item for sublist in lon_lats for item in sublist]
                mod_min = [item for sublist in mod_min for item in sublist]
                valid_cols_lon = [item for sublist in valid_cols_lon for item in sublist]

                if combine_pics:
                    ds_all, dates, masks, lon_lats, mod_min = combine_images_based_on_time(ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon)

                return ds_all, dates, masks, lon_lats, mod_min

            else:
                ds_all, dates, masks, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]
                return ds_all, dates, masks
    else:
        if save == None:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(
                    tqdm(
                        executor.map(
                            process_key, 
                            selected_keys, 
                            [file_groups] * len(selected_keys),
                            [file_layers] * len(selected_keys),
                            [bands] * len(selected_keys),
                            [water_mask_ravel] * len(selected_keys),
                            [tree] * len(selected_keys),
                            [mod_mins] * len(selected_keys),
                            [return_lon_lat] * len(selected_keys),
                            [max_zenith] * len(selected_keys)
                        ), 
                        total=len(selected_keys)
                    )
                )
            if return_lon_lat:
                ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]
                lon_lats = [item for sublist in lon_lats for item in sublist]
                mod_min = [item for sublist in mod_min for item in sublist]
                valid_cols_lon = [item for sublist in valid_cols_lon for item in sublist]


                if combine_pics:
                    ds_all, dates, masks, lon_lats, mod_min = combine_images_based_on_time(ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon)
                
                return ds_all, dates, masks, lon_lats, mod_min
            
            else:
                ds_all, dates, masks, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]

                return ds_all, dates, masks

def process_key(key, file_groups, file_layers, bands, full_water_mask, tree, mod_mins, return_lon_lat=False, max_zenith=50):
    file_group = file_groups[key]
    selected_mod_mins  = mod_mins[key]
    X = []
    dates = []
    masks = []
    lon_lats = []
    mod_min_list = []
    valid_cols_lon_list = []
    
    if return_lon_lat:
        for (file, mod_min) in zip(file_group, selected_mod_mins):
            result, mask, lon_lat, valid_cols_lon = process_file(file, file_layers, bands, full_water_mask, tree, return_lon_lat, max_zenith)

            if result.shape[0] > 1 and result.shape[1] > 1:
                X.append(result) 
                dates.append(key)
                masks.append(mask)
                lon_lats.append(lon_lat)
                mod_min_list.append(mod_min)
                valid_cols_lon_list.append(valid_cols_lon)
        
        return X, dates, masks, lon_lats, mod_min_list, valid_cols_lon_list

    else:
        for file in file_group:
            result, mask,valid_cols_lon = process_file(file, file_layers, bands, full_water_mask, tree, return_lon_lat, max_zenith)

            if  result.shape[0] > 1 and result.shape[1] > 1:
                X.append(result) 
                dates.append(key)
                masks.append(mask)
                valid_cols_lon_list.append(valid_cols_lon)

        return X, dates, masks, valid_cols_lon_list

    
def combine_images_based_on_time(ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon):
    combined_ds = []
    combined_dates = []
    combined_masks = []
    combined_lon_lats = []
    # Sort everything based on dates and mod_min
    sorted_indices = sorted(range(len(dates)), key=lambda x: (dates[x], mod_min[x]))
    ds_all = [ds_all[i] for i in sorted_indices]
    dates = [dates[i] for i in sorted_indices]
    masks = [masks[i] for i in sorted_indices]
    lon_lats = [lon_lats[i] for i in sorted_indices]
    mod_min = [mod_min[i] for i in sorted_indices]
    valid_cols_lon = [valid_cols_lon[i] for i in sorted_indices]
    combined_mod_min = []
   
    i = 0

    while i < len(dates) - 1:

        imgs_to_combine = [ds_all[i]]
        masks_to_combine = [masks[i]]
        lon_lats_to_combine = [lon_lats[i]]
        valid_cols_to_combine = [valid_cols_lon[i]]
        date = dates[i]
        min_time = mod_min[i]
        mod_min_start = min_time

        while i < len(dates) - 1 and ((mod_min[i+1] - min_time) == 5 or (mod_min[i+1] % 100 == 0 and ((min_time+45) % 100 == 0))):#or (min_time == 2355 and mod_min[i+1] == 0)):
            
            imgs_to_combine.append(ds_all[i+1])
            masks_to_combine.append(masks[i+1])
            lon_lats_to_combine.append(lon_lats[i+1])
            valid_cols_to_combine.append(valid_cols_lon[i+1])
            i += 1
            min_time = mod_min[i]

        # Find overlapping columns
        num_columns = imgs_to_combine[0].shape[1]

        # Initialize the mask as all True
        full_final_valid_cols_mask = np.ones(num_columns, dtype=bool)

        # Update the mask for each valid_cols
        for valid_cols in valid_cols_to_combine:
            full_final_valid_cols_mask &= valid_cols
        
        full_final_valid_cols_mask = np.any(valid_cols_to_combine, axis=0)

        # Use the mask to extract the overlapping columns
        combined_mod_min.append(mod_min_start)
        combined_ds.append(np.vstack([img[:, full_final_valid_cols_mask] for img in imgs_to_combine]))
        combined_dates.append(date)  # Taking the first date
        combined_masks.append(np.vstack([mask[:, full_final_valid_cols_mask] for mask in masks_to_combine]))
        combined_lon_lats.append(np.concatenate([ll[:,:, full_final_valid_cols_mask] for ll in lon_lats_to_combine], axis=1))
        
        i += 1

    print(len(combined_mod_min))
    print(len(combined_dates))

    # For the last image if it's standalone
    if len(imgs_to_combine) == 1:
        combined_mod_min.append(mod_min[-1])
        combined_ds.append(ds_all[-1][:, valid_cols_to_combine[0]]) 
        combined_masks.append(masks[-1][:, valid_cols_to_combine[0]]) 
        combined_lon_lats.append(lon_lats[-1][:,:, valid_cols_to_combine[0]]) 
        combined_dates.append(dates[-1])
        # combined_masks.append(masks[-1][valid_cols_to_combine[0]])
        # combined_lon_lats.append(lon_lats[-1][valid_cols_to_combine[0]])

    return combined_ds, combined_dates, combined_masks, combined_lon_lats, combined_mod_min



def extract_250m_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/", bands = [1,2],  save=None, start_date=None, end_date=None, date_list=None, min_mean=0, normalize=False):
    
    print("Preprocess")
    all_files = [f for f in os.listdir(folder) if f.endswith('.hdf')]
    hdf = SD(folder + all_files[0], SDC.READ)
    
    list1 = [int(num_str) for num_str in hdf.select("EV_250_RefSB").attributes()["band_names"].split(",")]
    file_layers = np.empty(2, dtype=object)
    for i, (band) in enumerate(list1):
        file_layers[band-1] = {"EV_250_RefSB": i}

    file_groups = defaultdict(list)

    # Loop through all files and group them by date
    for file in all_files:
        # Extract date from the filename (assuming the pattern is consistent)
        date = file.split('.')[1][1:]  # This will give e.g., '2021120' for 'MOD02QKM.A2021120'
        file_groups[date].append(file)


    sorted_keys = sorted(file_groups.keys(), key=int)  # Convert keys to integers for sorting

    # Extract the keys between start and end dates
    if start_date == None and end_date == None:
        selected_keys = [key for key in date_list if key in sorted_keys]
    else:
        selected_keys = [key for key in sorted_keys if int(start_date) <= int(key) <= int(end_date)]
    # with ProcessPoolExecutor() as executor:
    #     X = list(executor.map(append_data, [folder]*len(all_files), all_files, [file_layers]*len(all_files), [bands]*len(all_files)))

    ds_all = []
    if save == None:
        for key in list(selected_keys):
            file_group = file_groups[key]
            with ProcessPoolExecutor(max_workers=len(file_group)) as executor:
                X = list(tqdm(executor.map(append_data, [folder]*len(file_group), file_group, [file_layers]*len(file_group), [bands]*len(file_group), [min_mean]*len(file_group), [normalize]*len(file_group)), total=len(file_group)))

            ds_all.extend([xi for xi in X if xi.ndim>1])

        return ds_all
    else:
        for key in file_groups.keys():
            if not os.path.exists(save +"_" + key +".npz"):
                file_group = file_groups[key]
                with ProcessPoolExecutor(max_workers=len(file_group)) as executor:
                    X = list(tqdm(executor.map(append_data, [folder]*len(file_group), file_group, [file_layers]*len(file_group), [bands]*len(file_group), [min_mean]*len(file_group)), total=len(file_group)))
                
                if X.ndims > 1:
                     ds_all.extend(X)

                if save != None:
                    print("Saving...")
                    np.savez(save +"_" + key, *ds_all)
            else:
                print("exists")
            gc.collect()

    

def replace_out_of_bounds_with_nearest(data, low_bound, high_bound):
    out_of_bounds = ~((data >= low_bound) & (data <= high_bound))

    if np.any(out_of_bounds):

        # Compute the distances to the nearest valid value
        distances, (i, j) = distance_transform_edt(out_of_bounds, return_indices=True)
        
        # Use the indices to get the values from the nearest valid value
        data[out_of_bounds] = data[i[out_of_bounds], j[out_of_bounds]]

    return data


def process_file(file, file_layers, bands, full_water_mask=None, tree=None, return_lon_lat=False, max_zenith=50):
    if file.endswith(".hdf"):
        return process_hdf_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat)
    elif file.endswith(".npy"):
        return process_npy_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat)
    else:
        raise ValueError(f"Unsupported file format: {file}")

def process_hdf_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat):
    zenith = np.load("%sland_sea_ice_mask/sensor_zenith_bilinear_1km.npy" %data_loc)
    zenith_mask = zenith < max_zenith
    current_data_list = []

    hdf = SD(file, SDC.READ)

    #### IF FILE LAYER IS A STRING DO NOT INDEX ....
    key = list(file_layers[bands[0]-1].keys())[0]
    idx = list(file_layers[bands[0]-1].values())[0]
    data = hdf.select(key)[:][idx]
    lat = hdf.select("Latitude")[:]
    lon = hdf.select("Longitude")[:]

    lat = replace_out_of_bounds_with_nearest(lat, -90, 90)
    lon = replace_out_of_bounds_with_nearest(lon, -180, 180)

    lon_min, lon_max = -35, 45
    lat_min, lat_max = 60, 82
    
    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon > lon_min) & (lon < lon_max)
    zoom_factor_y = data.shape[0] / lat.shape[0]
    zoom_factor_x = data.shape[1] / lat.shape[1]
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions
    

    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)

    valid_cols_ll = zenith_mask
    valid_cols_lon = np.any(mask_highres[valid_rows_ll][:,valid_cols_ll] , axis=0)  ### remove last for matching x-axis
    data = data[valid_rows_ll][:, valid_cols_ll]
    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 

    #coords_lowres = np.column_stack((full_water_mask.latitude.values.ravel(), full_water_mask.longitude.values.ravel()))

    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    mask = full_water_mask[indices].reshape(data.shape)
    attrs = hdf.select(key).attributes()
    is_nan = data == attrs["_FillValue"]
       
    valid_rows = ~np.all(is_nan, axis=1)
    valid_cols = ~np.all(is_nan, axis=0)
    data = data[valid_rows][:, valid_cols]
    mask = mask[valid_rows][:, valid_cols]
    lon_highres = lon_highres[valid_rows][:, valid_cols]
    lat_highres = lat_highres[valid_rows][:, valid_cols]
    


    data_shape_bool = data.shape[0] !=0 and data.shape[1] != 0
    if data_shape_bool:
        data = np.where(data > attrs["valid_range"][1], np.mean(data), data)
        data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
        current_data_list.append(data)
    else:
        current_data_list.append(np.empty((0,0)))

    for j, (band) in enumerate(bands[1:]):
        key = list(file_layers[band-1].keys())[0]
        idx = list(file_layers[band-1].values())[0]

        attrs = hdf.select(key).attributes()
        data = hdf.select(key)[:][idx]
        data = data[valid_rows_ll][:, valid_cols_ll]

        data = data[valid_rows][:, valid_cols]
        data = np.where(data > attrs["valid_range"][1], np.mean(data), data)

        data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
                    
        current_data_list.append(data)
    
    
    ##### ALGORITHM LOOKS ONLY AT NEAREST LAT LON AND NOT THE EXACT DISTANCE IN DETERMINATION   
    x_bands = np.stack(current_data_list, axis=-1)
   
    if return_lon_lat:
        return x_bands, mask, np.array([lon_highres, lat_highres]), valid_cols_lon 
    else:
        return x_bands, mask, valid_cols_lon 

def process_npy_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat):
    zenith = np.load("%sland_sea_ice_mask/sensor_zenith_bilinear_1km.npy" %data_loc)
    zenith_mask = zenith < max_zenith
    current_data_list = []
    ds = np.load(file, allow_pickle=True).item()
    data = ds["data"]
    lon = ds["lon"]
    lat = ds["lat"]

    lat = replace_out_of_bounds_with_nearest(lat, -90, 90)
    lon = replace_out_of_bounds_with_nearest(lon, -180, 180)

    # Inside region of interest check
    lon_min, lon_max = -35, 45
    lat_min, lat_max = 60, 82
    
    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    zoom_factor_y = data.shape[0] / lat.shape[0]
    zoom_factor_x = data.shape[1] / lat.shape[1]
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions
    

    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)

    valid_cols_ll = zenith_mask
    valid_cols_lon = np.any(mask_highres[valid_rows_ll][:,valid_cols_ll] , axis=0)  ### remove last for matching x-axis
    data = data[valid_rows_ll][:, valid_cols_ll]
    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 


    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    mask = full_water_mask[indices].reshape(data.shape)

    # nan check of whole rows and columns
  # Check for columns that are all NaN
    is_nan = np.isnan(data)
    valid_cols = ~np.all(is_nan, axis=0)

    # If any invalid columns exist, create an empty dataset
    # if not np.all(valid_cols):
    #     print("FOUND UNVALID DATA")
    #     data = np.empty((0, 0))
    # else:
    # Check for rows that are all NaN
    valid_rows = ~np.all(is_nan, axis=1)

    # Use only valid data
    data = data[valid_rows][:, valid_cols]
    mask = mask[valid_rows][:, valid_cols]
    lon_highres = lon_highres[valid_rows][:, valid_cols]
    lat_highres = lat_highres[valid_rows][:, valid_cols]

    # nan check of single values inside dataset
    is_nan = np.isnan(data)
    data_shape_bool = data.shape[0] !=0 and data.shape[1] != 0

    if data_shape_bool:
        data = np.where(is_nan, np.nanmean(data), data)
    else:
        data = np.empty((0, 0))

    current_data_list.append(data)
    
    ##### ALGORITHM LOOKS ONLY AT NEAREST LAT LON AND NOT THE EXACT DISTANCE IN DETERMINATION   
    x_bands = np.stack(current_data_list, axis=-1)

    if return_lon_lat:
        return x_bands, mask, np.array([lon_highres, lat_highres]), valid_cols_lon 
    else:
        return x_bands, mask, valid_cols_lon


def append_data(file, file_layers, bands, min_mean=0, full_water_mask=None, tree=None, return_lon_lat=False, normalize=False, max_zenith=50):
    zenith = np.load("%sland_sea_ice_mask/sensor_zenith_bilinear_1km.npy" %data_loc)
    zenith_mask = zenith < max_zenith
    current_data_list = []

    if file.endswith(".hdf"):
        hdf = SD(file, SDC.READ)
        key = list(file_layers[bands[0]-1].keys())[0]
        idx = list(file_layers[bands[0]-1].values())[0]
        data = hdf.select(key)[:][idx]
        lat = hdf.select("Latitude")[:]
        lon = hdf.select("Longitude")[:]
    elif file.endswith(".npy"):
        ds = np.load(file, allow_pickle=True).item()
        data = ds["data"]
        lon = ds["lon"]
        lat = ds["lat"]

    lat = replace_out_of_bounds_with_nearest(lat, -90, 90)
    lon = replace_out_of_bounds_with_nearest(lon, -180, 180)

    lon_min, lon_max = -35, 45
    lat_min, lat_max = 60, 82
    
    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon > lon_min) & (lon < lon_max)
    zoom_factor_y = data.shape[0] / lat.shape[0]
    zoom_factor_x = data.shape[1] / lat.shape[1]
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions
    

    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)

    valid_cols_ll = zenith_mask
    valid_cols_lon = np.any(mask_highres[:][:,valid_cols_ll] , axis=0)  ### remove last for matching x-axis
    print(valid_cols_lon.shape)
    data = data[valid_rows_ll][:, valid_cols_ll]
    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 

    #coords_lowres = np.column_stack((full_water_mask.latitude.values.ravel(), full_water_mask.longitude.values.ravel()))

    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    mask = full_water_mask[indices].reshape(data.shape)
    if file.endswith(".hdf"):
        attrs = hdf.select(key).attributes()
        is_nan = data == attrs["_FillValue"]
    elif file.endswith(".npy"):
        is_nan = np.isnan(data)

    valid_rows = ~np.all(is_nan, axis=1)
    valid_cols = ~np.all(is_nan, axis=0)
    data = data[valid_rows][:, valid_cols]
    mask = mask[valid_rows][:, valid_cols]
    lon_highres = lon_highres[valid_rows][:, valid_cols]
    lat_highres = lat_highres[valid_rows][:, valid_cols]
    
    data_shape_bool = data.shape[0] !=0 and data.shape[1] != 0
    if data_shape_bool:
        if file.endswith(".hdf"):
            data = np.where(data > attrs["valid_range"][1], 0, data)
            data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
        elif file.endswith(".npy"):
            data = np.where(is_nan, np.nanmean(data), data)
        current_data_list.append(data)
    else:
        current_data_list.append(np.empty((0,0)))

    if file.endswith(".hdf"):
        for j, (band) in enumerate(bands[1:]):
            key = list(file_layers[band-1].keys())[0]
            idx = list(file_layers[band-1].values())[0]

            attrs = hdf.select(key).attributes()
            data = hdf.select(key)[:][idx]
            data = data[valid_rows_ll][:, valid_cols_ll]

            data = data[valid_rows][:, valid_cols]
            data = np.where(data > attrs["valid_range"][1], 0, data)

            data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
                        
            current_data_list.append(data)
    
    
    ##### ALGORITHM LOOKS ONLY AT NEAREST LAT LON AND NOT THE EXACT DISTANCE IN DETERMINATION   
    x_bands = np.stack(current_data_list, axis=-1)
   
    if return_lon_lat:
        return x_bands, mask, np.array([lon_highres, lat_highres]), valid_cols_lon 
    else:
        return x_bands, mask, valid_cols_lon 
    

def normalize_data(data):
    normalized_data = []    
    normalized_data.append((data - np.nanmin(data, axis=(0,1), keepdims=True)) / (np.nanmax(data, axis=(0,1), keepdims=True) - np.nanmin(data, axis=(0,1), keepdims=True)))
    #normalized_data = (data - np.nanmin(data, axis=(1,2), keepdims=True)) / (np.nanmax(data, axis=(1,2), keepdims=True) - np.nanmin(data, axis=(1,2), keepdims=True))
    return normalized_data
    
# import time 
# start = time.time()
# print(os.cpu_count())
# #folder = "/uio/hume/student-u37/fslippe/data/nird_mount/"
# folder = "/nird/projects/NS9600K/data/modis/cao/"

# #x = extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1,2],  save=None)
# #extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1, 2],  save=folder + "MOD02QKM_202012-202104/training_set")
# #extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1, 2],  save= folder + "MOD02QKM_202012-202104/converted_data/training_set")
# extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1, 2],  save= folder + "MOD02QKM_202012-202104/converted_data/training_set")

# #loaded = np.load("/nird/projects/NS9600K/fslippe/test.npz")
# #X = [loaded[key] for key in loaded]
# #print(X)
# end = time.time()

# print("time used:", end-start)
