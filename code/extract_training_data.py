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


total_cores = multiprocessing.cpu_count()
print("total cores:", total_cores)

def extract_1km_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/",
                     bands = [6,7,20,28,28,31],
                     ds_water_mask=xr.open_dataset("/uio/hume/student-u37/fslippe/data/land_sea_ice_mask/sea_land_mask.nc"),
                     save=None,
                     start_date=None,
                     end_date=None,
                     date_list=None,
                     min_mean=0,
                     normalize=None,
                     return_lon_lat=False,
                     workers=None,
                     max_zenith=50):
    
    all_files = []
    folders = folder.split(" ")
    print(folders)

    for f in folders:
        all_files.extend([os.path.join(f, file) for file in os.listdir(f) if file.endswith('.hdf')])
    
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
    
    ##### FOLLOWING CODE IS USED IF PARALLELIZING all files
    if workers == None:
        if len(selected_keys) < 10:
            workers = len(selected_keys)
        else:
            workers = 128
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
                        [min_mean] * len(selected_keys),
                        [ds_water_mask] * len(selected_keys),
                        [normalize] * len(selected_keys),
                        [return_lon_lat] * len(selected_keys),
                        [max_zenith] * len(selected_keys)
                    ), 
                    total=len(selected_keys)
                )
            )
        if return_lon_lat:
            ds_all, dates, masks, lon_lats = zip(*results)
            ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
            dates = [item for sublist in dates for item in sublist]
            masks = [item for sublist in masks for item in sublist]
            lon_lats = [item for sublist in lon_lats for item in sublist]
            return ds_all, dates, masks, lon_lats
        
        else:
            ds_all, dates, masks = zip(*results)
            ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
            dates = [item for sublist in dates for item in sublist]
            masks = [item for sublist in masks for item in sublist]
            return ds_all, dates, masks

def process_key(key, file_groups, file_layers, bands, min_mean, full_water_mask, normalize, return_lon_lat=False, max_zenith=50):
    file_group = file_groups[key]
    X = []
    dates = []
    masks = []
    lon_lats = []

    if return_lon_lat:
        for file in file_group:
            result, mask, lon_lat = append_data(file, file_layers, bands, min_mean, full_water_mask, return_lon_lat, normalize, max_zenith)

            if result.ndim > 1:
                X.append(result) 
                dates.append(key)
                masks.append(mask)
                lon_lats.append(lon_lat)

        return X, dates, masks, lon_lats

    else:
        for file in file_group:
            result, mask = append_data(file, file_layers, bands, min_mean, full_water_mask, return_lon_lat, normalize, max_zenith)

            if result.ndim > 1:
                X.append(result) 
                dates.append(key)
                masks.append(mask)
        return X, dates, masks




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
            print("Date:", convert_to_standard_date(key))
            with ProcessPoolExecutor(max_workers=len(file_group)) as executor:
                X = list(tqdm(executor.map(append_data, [folder]*len(file_group), file_group, [file_layers]*len(file_group), [bands]*len(file_group), [min_mean]*len(file_group), [normalize]*len(file_group)), total=len(file_group)))

            ds_all.extend([xi for xi in X if xi.ndim>1])

        return ds_all
    else:
        for key in file_groups.keys():
            if not os.path.exists(save +"_" + key +".npz"):
                file_group = file_groups[key]
                print(file_group)
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

    


def append_data(file, file_layers, bands, min_mean=0, full_water_mask=None, return_lon_lat=False, normalize=False, max_zenith=50):
    zenith = np.load("/uio/hume/student-u37/fslippe/master_project/code/data/sensor_zenith_bilinear_1km.npy")
    zenith_mask = zenith < max_zenith
    hdf = SD(file, SDC.READ)
    current_data_list = []
    key = list(file_layers[bands[0]-1].keys())[0]
    idx = list(file_layers[bands[0]-1].values())[0]
    data = hdf.select(key)[:][idx]
    lat = hdf.select("Latitude")[:]
    lon = hdf.select("Longitude")[:]

    lon_min, lon_max = -35, 35
    lat_min, lat_max = 60, 82


    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon > lon_min) & (lon < lon_max)
    zoom_factor_y = data.shape[0] / lat.shape[0]
    zoom_factor_x = data.shape[1] / lat.shape[1]
    
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions

    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # order=0 for nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # order=1 for bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)
    valid_cols_ll = np.any(mask_highres , axis=0) & zenith_mask
    data = data[valid_rows_ll][:, valid_cols_ll]
    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 

    coords_lowres = np.column_stack((full_water_mask.latitude.values.ravel(), full_water_mask.longitude.values.ravel()))
    tree = cKDTree(coords_lowres)
    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1)
    mask = full_water_mask["sea_ice_region_surface_mask"].values.ravel()[indices].reshape(data.shape)
    
    attrs = hdf.select(key).attributes()
    is_nan = data == attrs["_FillValue"]
    valid_rows = ~np.all(is_nan, axis=1)
    valid_cols = ~np.all(is_nan, axis=0)
    data = data[valid_rows][:, valid_cols]

    data = np.where(data > attrs["valid_range"][1], 0, data)
    data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])

    current_data_list.append(data)
    
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
        return x_bands, mask, np.array([lon_highres, lat_highres]) 
    else:
        return x_bands, mask
    

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
