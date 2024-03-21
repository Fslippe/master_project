import  numpy as np  
import os 

years = [2020, 2021, 2022, 2023]
for year in years:
    folder = f"/scratch/fslippe/modis/MOD02_npy/{year}/"
    files = os.listdir(folder)
    file = np.load(folder + files[0], allow_pickle=True).item()


    for filename in files:
        # Load the data from the original .npy file
        file_path = os.path.join(folder, filename)
        data = np.load(file_path, allow_pickle=True).item()
        
        # Extract data arrays from the dictionary
        lon = data['lon']
        lat = data['lat']
        data_array = data['data']
        
        # Define the destination path for the compressed file
        # It is advised to use a different folder to avoid filename conflicts
        compressed_folder = f"/scratch/fslippe/modis/MOD02_npz/{year}/"
        if not os.path.exists(compressed_folder):
            os.makedirs(compressed_folder)
        compressed_file_path = os.path.join(compressed_folder, filename.replace('.npy', '.npz'))
        
        # Save the arrays in one compressed .npz file
        np.savez_compressed(compressed_file_path, lon=lon, lat=lat, data=data_array)
