import os

# Provide the directory path where the files are located
for n_k in [10, 11, 12, 13, 14, 15, 16]:
    directory = f'/uio/hume/student-u37/fslippe/data/models/patch_size128/filter64/clustering/cao_date_time_lists/n_K_{n_k}/'

    # Get all the files in the directory
    files = os.listdir(directory)

    # Iterate through each file in the directory
    for file in files:
        if file.endswith(".npy"):
            # Generate the new file name by adding "_2019" before ".npy"
            new_file_name = file.replace("__2019.npy", "_2019.npy")
            
            # Rename the file
            os.rename(os.path.join(directory, file), os.path.join(directory, new_file_name))


