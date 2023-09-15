import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

def process_key(args):
    loaded, key = args
    normalized_data_dict = {}
    
    data = loaded[key]
    if data.shape[0] >= 256:
        print(data.shape[2])
        if data.shape[2] == 3:  # Check if the third dimension is of size 3
            data = data[:,:,1:]
        normalized_data = normalize_data(data)
        normalized_data_dict[key] = normalized_data
    
    return normalized_data_dict


def normalize_data(data):
    return (data - np.min(data, axis=(0,1), keepdims=True)) / (np.max(data, axis=(0,1), keepdims=True) - np.min(data, axis=(0,1), keepdims=True))

from concurrent.futures import ProcessPoolExecutor

folder = "/nird/projects/NS9600K/data/modis/cao/MOD02QKM_202012-202104/converted_data"
all_files = [f for f in os.listdir(folder) if f.endswith('.npz')]

def process_file(file):
    save_path = "/nird/projects/NS9600K/data/modis/cao/MOD02QKM_202012-202104/normalized_data/" + "normalized_" + file  
    if not os.path.exists(save_path + ".npz"):
        filepath = folder + "/" + file
        loaded = np.load(filepath, allow_pickle=True)

        normalized_data_dict = {}
        for key in loaded:
            normalized_data_dict.update(process_key((loaded, key)))

        np.savez(save_path, **normalized_data_dict)

with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
    executor.map(process_file, all_files)



# for file in all_files:
#     save_path = "/nird/projects/NS9600K/data/modis/cao/MOD02QKM_202012-202104/normalized_data/normalized_" + file
    
#     # Only process the file if the normalized version does not exist yet
#     if not os.path.exists(save_path + ".npz"):
#         print(f"Processing {file}")
#         filepath = os.path.join(folder, file)
#         loaded = np.load(filepath, allow_pickle=True)  # allow_pickle might be needed for some npz files
        
#         # Create a dictionary to hold normalized arrays
#         normalized_data_dict = {}

#         # Process each key in sequence
#         start_time = time.time()
#         for key in loaded:
#             normalized_data_dict.update(process_key((loaded, key)))
            
#         print("Time taken:", time.time() - start_time)
#         print("Number of entries:", len(normalized_data_dict))

#         np.savez(save_path, **normalized_data_dict)
