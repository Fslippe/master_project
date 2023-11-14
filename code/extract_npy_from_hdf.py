import os
import numpy as np
from pyhdf.SD import SD, SDC
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_all_files_in_folders(folders):
    all_files = []
    for folder in folders:
        all_files.extend([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.hdf')])
    return all_files

def process_hdf_file(file, key, idx, band, attrs):
    filename = file.split("/")[-1][:-4]
    year = filename.split(".")[1][1:5]
    file_loc = f"/scratch/fslippe/modis/MOD02/{year}/{filename}_ll_band_{band}.npy"

    if not os.path.exists(file_loc):
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

        np.save(file_loc, data_dict)

def process_files_parallel(files, key, idx, band, attrs):
    with ThreadPoolExecutor(max_workers=32) as executor, tqdm(total=len(files), desc="Processing Files") as pbar:
        for _ in executor.map(lambda file: process_hdf_file(file, key, idx, band, attrs), files):
            pbar.update(1)

def process_files_serial(files, key, idx, band, attrs):
    with tqdm(total=len(files), desc="Processing Files") as pbar:
        for file in files:
            process_hdf_file(file, key, idx, band, attrs)
            pbar.update(1)

def main():
    # folders = ["/scratch/fslippe/modis/MOD02/daytime_1km/",
    #            "/scratch/fslippe/modis/MOD02/boundary_1km/",
    #            "/scratch/fslippe/modis/MOD02/night_1km/",
    #            "/scratch/fslippe/modis/MOD02/may-nov_2021/",
    #            "/scratch/fslippe/modis/MOD02/cao_test_data/"]
    folders = ["/uio/hume/student-u37/fslippe/nird_mod02/2020/"]

    all_files = get_all_files_in_folders(folders)#[1000:]
    
    key = "EV_1KM_Emissive"
    idx = 4
    band = 29
    
    # Assuming attrs are the same for all files, using the first file to get attributes
    hdf_attrs = SD(all_files[0], SDC.READ) 
    attrs = hdf_attrs.select(key).attributes()

    process_files_serial(all_files, key, idx, band, attrs)

if __name__ == "__main__":
    main()
