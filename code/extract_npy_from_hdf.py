import os
import numpy as np
from pyhdf.SD import SD, SDC
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def get_all_files_in_folders(folders):
    all_files = []
    for folder in folders:
        all_files.extend([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.hdf')])
    return all_files

def process_hdf_file(file, key, idx, band, attrs, base_save_folder="/scratch/fslippe/modis/MOD02/"):
    filename = file.split("/")[-1][:-4]
    year = filename.split(".")[1][1:5]
    file_loc = f"{base_save_folder}{year}/{filename}_ll_band_{band}.npz"

    if not os.path.exists(file_loc):
        try:
            hdf = SD(file, SDC.READ)
            data = hdf.select(key)[:][idx]
            data = np.where(data == attrs["_FillValue"], np.nan, data)
            out_of_range = np.where(data > attrs["valid_range"][1])
            data = np.float32((data - attrs["radiance_offsets"][idx]) * attrs["radiance_scales"][idx])
            data[out_of_range] = np.nan
            lat = hdf.select("Latitude")[:]
            lon = hdf.select("Longitude")[:]

            data_dict = {
                'lon': lon,
                'lat': lat,
                'data': data
            }
            np.savez_compressed(file_loc, lon=lon, lat=lat, data=data)

            #np.save(file_loc, data_dict)
        except:
            print("FAILED ON HDF FILE", file)
        
from functools import partial

def process_func(file, key, idx, band, attrs, base_save_folder):
    return process_hdf_file(file, key, idx, band, attrs, base_save_folder)



def process_files_parallel(files, key, idx, band, attrs, base_save_folder, workers=4):
    process_partial = partial(process_func, key=key, idx=idx, band=band, attrs=attrs, base_save_folder=base_save_folder)
    for _ in executor.map(process_partial, files):
        pbar.update(1)
    # with ProcessPoolExecutor(max_workers=workers) as executor, tqdm(total=len(files), desc="Processing Files") as pbar:
    #     for _ in executor.map(lambda file: process_hdf_file(file, key, idx, band, attrs, base_save_folder), files):
    #         pbar.update(1)

def process_files_serial(files, key, idx, band, attrs, base_save_folder):
    with tqdm(total=len(files), desc="Processing Files") as pbar:
        for file in files:
            process_hdf_file(file, key, idx, band, attrs, base_save_folder)
            pbar.update(1)

def main():
    folders = ["/scratch/fslippe/modis/MOD02/daytime_1km/",
               "/scratch/fslippe/modis/MOD02/boundary_1km/",
               "/scratch/fslippe/modis/MOD02/night_1km/",
               "/scratch/fslippe/modis/MOD02/may-nov_2021/",
               "/scratch/fslippe/modis/MOD02/cao_test_data/"]
    folders = ["/scratch/fslippe/modis/MOD02/2023/"]

    base_save_folder = "/scratch/fslippe/modis/MOD02_npz/"
    all_files = get_all_files_in_folders(folders)[:]#[1000:]
    length = (len(all_files))
    print(length)
    #all_files_2 = all_files[length // 2:]
    #all_files = all_files[:length // 2]
    #print(len(all_files_2), len(all_files)) 
    key = "EV_1KM_Emissive"
    band = 29
    hdf_attrs = SD(all_files[0], SDC.READ) 
    attrs = hdf_attrs.select(key).attributes()
    idx = (np.where(np.array(attrs["band_names"].split(",")) == "%s" %band)[0][0])
    # Assuming attrs are the same for all files, using the first file to get attributes
    
    process_files_serial(all_files, key, idx, band, attrs, base_save_folder)
    #process_files_parallel(all_files, key, idx, band, attrs, base_save_folder, workers = 8)


if __name__ == "__main__":
    main()

