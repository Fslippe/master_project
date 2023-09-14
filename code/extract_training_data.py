from pyhdf.SD import SD, SDC
import numpy as np 
import os 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

def extract_250m_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/winter_202012-202004/", bands = [1,2],  save=None):
    print("Preprocess")
    all_files = [f for f in os.listdir(folder) if f.endswith('.hdf')][:50]

    print(folder + all_files[0])
    hdf = SD(folder + all_files[0], SDC.READ)
    datasets = hdf.datasets()
    for idx, sds in enumerate(datasets.keys()):
        print(sds, hdf.select(sds).attributes())
        
    list1 = [int(num_str) for num_str in hdf.select("EV_250_RefSB").attributes()["band_names"].split(",")]

    file_layers = np.empty(2, dtype=object)
    for i, (band) in enumerate(list1):
        file_layers[band-1] = {"EV_250_RefSB": i}


    #all_files = os.listdir(folder)[16:18]
    print(len(all_files))
    
    #X = np.empty((len(all_files), 2030, 1354, len(bands)))
    print("Importing files to RAM")

    # with ProcessPoolExecutor() as executor:
    #     X = list(executor.map(append_data, [folder]*len(all_files), all_files, [file_layers]*len(all_files), [bands]*len(all_files)))

    with ProcessPoolExecutor() as executor:
        X = list(tqdm(executor.map(append_data, [folder]*len(all_files), all_files, [file_layers]*len(all_files), [bands]*len(all_files)), total=len(all_files)))


    
    if save != None:
        print("Saving...")
        np.savez(save, *X)
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
    data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]
    current_data_list.append(data)
    
    for j, (band) in enumerate(bands):
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
#x = extract_250m_data(folder="/uio/hume/student-u37/fslippe/data/nird_mount/MOD02QKM_202012-202104/", bands = [1,2],  save=None)
extract_250m_data(folder="/nird/projects/NS9600K/data/modis/cao/MOD02QKM_202012-202104/", bands = [1, 2],  save="/nird/projects/NS9600K/data/modis/cao/MOD02QKM_202012-202104/training_set")
#loaded = np.load("/nird/projects/NS9600K/fslippe/test.npz")
#X = [loaded[key] for key in loaded]
#print(X)
end = time.time()
print("time used:", end-start)
# def append_data(folder, all_files, file_layers, bands):
#     X = []
#     for i, (file) in enumerate(all_files):
#         hdf = SD(folder + file, SDC.READ)
#         current_data_list = []

#         key = list(file_layers[bands[0]-1].keys())[0]
#         idx = list(file_layers[bands[0]-1].values())[0]
#         attrs = hdf.select(key).attributes()
#         data = hdf.select(key)[:][idx]
#         is_nan = data == attrs["_FillValue"]
#         valid_rows = ~np.all(is_nan, axis=1)
#         valid_cols = ~np.all(is_nan, axis=0)
#         data = data[valid_rows][:, valid_cols]
#         data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]
#         current_data_list.append(data)
        
#         for j, (band) in enumerate(bands[1:]):
#             key = list(file_layers[band-1].keys())[0]
#             idx = list(file_layers[band-1].values())[0]

#             attrs = hdf.select(key).attributes()
#             data = hdf.select(key)[:][idx]
#             data = data[valid_rows][:, valid_cols]

#             data = (data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx]
#             # if not len(is_nan[0]) == 0:
#             #     data = data[is_nan[0][-1]+1:, :] if is_nan[1][-1] == 1353 else data[:, is_nan[1][-1]+1:]
#             current_data_list.append(data)
#         x_bands = np.stack(current_data_list, axis=-1)
#         #    x_bands[:,:, j] = data
        
#         X.append(x_bands)
#     return X

