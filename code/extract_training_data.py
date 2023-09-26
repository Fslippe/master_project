from pyhdf.SD import SD, SDC
import numpy as np 
import os 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import gc
from functions import *



def extract_1km_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/", bands = [6, 7, 20, 28, 28, 31],  save=None, start_date=None, end_date=None, date_list=None, min_mean=0, normalize=None):

    all_files = [f for f in os.listdir(folder) if f.endswith('.hdf')]

    print(folder + all_files[0])
    hdf = SD(folder + all_files[0], SDC.READ)
    datasets = hdf.datasets()
    for idx, sds in enumerate(datasets.keys()):
        print(idx, sds)
        
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
    if len(selected_keys) < 10:
        workers = len(selected_keys)
    else:
        workers = 256
    if save == None:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            ds_all = list(
                tqdm(
                    executor.map(
                        process_key, 
                        selected_keys, 
                        [file_groups] * len(selected_keys),
                        [folder] * len(selected_keys),
                        [file_layers] * len(selected_keys),
                        [bands] * len(selected_keys),
                        [min_mean] * len(selected_keys),
                        [normalize] * len(selected_keys)
                    ), 
                    total=len(selected_keys)
                )
            )
        ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
        return ds_all

def process_key(key, file_groups, folder, file_layers, bands, min_mean, normalize):
    file_group = file_groups[key]
    X = []
    for file in file_group:
        result = append_data(folder, file, file_layers, bands, min_mean, normalize)
        X.append(result)
    return [xi for xi in X if xi.ndim > 1]



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

    


def append_data(folder, file, file_layers, bands, min_mean=0, normalize=False):
    hdf = SD(folder + file, SDC.READ)
    current_data_list = []

    key = list(file_layers[bands[0]-1].keys())[0]
    idx = list(file_layers[bands[0]-1].values())[0]
    attrs = hdf.select(key).attributes()
    data = hdf.select(key)[:][idx]
    is_nan = data == attrs["_FillValue"]
    valid_rows = ~np.all(is_nan, axis=1)
    valid_cols = ~np.all(is_nan, axis=0)
    data = data[valid_rows][:, valid_cols]
    #print(idx, np.mean(data), "\n\n")

    data = np.where(data > attrs["valid_range"][1], 0, data)
    data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
    #print(np.mean(data))

    if np.mean(data) > min_mean:
        current_data_list.append(data)
        
        for j, (band) in enumerate(bands[1:]):

            key = list(file_layers[band-1].keys())[0]
            idx = list(file_layers[band-1].values())[0]

            attrs = hdf.select(key).attributes()
            data = hdf.select(key)[:][idx]
            data = data[valid_rows][:, valid_cols]
            data = np.where(data > attrs["valid_range"][1], 0, data)

            data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]
                        
            current_data_list.append(data)
            
        x_bands = np.stack(current_data_list, axis=-1)
        return x_bands
    else:
        return np.empty(1)
    
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
