from pyhdf.SD import SD, SDC
import numpy as np 
import os 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import gc


def extract_1km_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/", bands = [6, 7, 20, 28, 28, 31],  save=None):

    #folder = "/nird/projects/NS9600K/data/modis/cao/"
    #folder = "/home/filip/Documents/master_project/data/MOD02/"
    #folder = "/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/"
    #folder = "/uio/hume/student-u37/fslippe/data/cao/"

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


    #all_files = os.listdir(folder)[16:18]
    print(len(all_files))

    #X = np.empty((len(all_files), 2030, 1354, len(bands)))
    with ProcessPoolExecutor(max_workers=4) as executor:
        X = list(executor.map(append_data, [folder]*len(all_files), all_files, [file_layers]*len(all_files), [bands]*len(all_files)))

    
    if save != None:
        np.savez('/uio/hume/student-u37/fslippe/data/training_data/training_data_20210421.npz', *X)
    else:
        return X

def extract_250m_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/", bands = [1,2],  save=None, workers=1):
    print("Preprocess")
    all_files = [f for f in os.listdir(folder) if f.endswith('.hdf')]
    hdf = SD(folder + all_files[0], SDC.READ)
    # datasets = hdf.datasets()
    # for idx, sds in enumerate(datasets.keys()):
    #     print(sds, hdf.select(sds).attributes())
        
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
        # print(folder + all_files[0])




    #all_files = os.listdir(folder)[16:18]
    print(len(all_files))
    
    #X = np.empty((len(all_files), 2030, 1354, len(bands)))
    print("Importing files to RAM")

    # with ProcessPoolExecutor() as executor:
    #     X = list(executor.map(append_data, [folder]*len(all_files), all_files, [file_layers]*len(all_files), [bands]*len(all_files)))
    for key in file_groups.keys():
        if not os.path.exists(save +"_" + key +".npz"):
            file_group = file_groups[key]
            print(file_group)
            with ProcessPoolExecutor(max_workers=len(file_group)) as executor:
                X = list(tqdm(executor.map(append_data, [folder]*len(file_group), file_group, [file_layers]*len(file_group), [bands]*len(file_group)), total=len(file_group)))

            if save != None:
                print("Saving...")
                np.savez(save +"_" + key, *X)
        else:
            print("exists")
        gc.collect()
    else:
        return X
    


def append_data(folder, file, file_layers, bands):
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
    print(data.shape[0])
    #if data.shape[0] < 64:
        
    data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]
    current_data_list.append(data)
    
    for j, (band) in enumerate(bands[1:]):
        key = list(file_layers[band-1].keys())[0]
        idx = list(file_layers[band-1].values())[0]

        attrs = hdf.select(key).attributes()
        data = hdf.select(key)[:][idx]
        data = data[valid_rows][:, valid_cols]
        data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]
        current_data_list.append(data)
        
    x_bands = np.stack(current_data_list, axis=-1)
    return x_bands

import time 
start = time.time()
print(os.cpu_count())
#folder = "/uio/hume/student-u37/fslippe/data/nird_mount/"
folder = "/nird/projects/NS9600K/data/modis/cao/"

#x = extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1,2],  save=None)
#extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1, 2],  save=folder + "MOD02QKM_202012-202104/training_set")
#extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1, 2],  save= folder + "MOD02QKM_202012-202104/converted_data/training_set")
extract_250m_data(folder=folder + "MOD02QKM_202012-202104/", bands = [1, 2],  save= folder + "MOD02QKM_202012-202104/converted_data/training_set")

#loaded = np.load("/nird/projects/NS9600K/fslippe/test.npz")
#X = [loaded[key] for key in loaded]
#print(X)
end = time.time()

print("time used:", end-start)
