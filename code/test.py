import os

# Provide the directory path where the files are located
# for n_k in [10, 11, 12, 13, 14, 15, 16]:
#     directory = f'/uio/hume/student-u37/fslippe/data/models/patch_size128/filter64/clustering/cao_date_time_lists/n_K_{n_k}/'

#     # Get all the files in the directory
#     files = os.listdir(directory)

#     # Iterate through each file in the directory
#     for file in files:
#         if file.endswith(".npy"):
#             # Generate the new file name by adding "_2019" before ".npy"
#             new_file_name = file.replace("__2019.npy", "_2019.npy")
            
#             # Rename the file
#             os.rename(os.path.join(directory, file), os.path.join(directory, new_file_name))


import os

wildcards = ["2019001","2019002","2019013","2019014","2019017","2019018","2019027","2019048","2019049","2019050","2019060","2019062","2019063","2019064","2019065","2019067","2019081","2019089","2019295","2019296","2019333","2019342","2019343","2020004","2020023","2020043","2020050","2020054","2020057","2020058","2020060","2020072","2020073","2020074","2020079","2020080","2020087","2020088","2020093","2020094","2020095","2020362","2021020","2021021","2021022","2021024","2021040","2021097","2021327","2021328","2021331","2021332","2021333","2022004","2022005","2022015","2022016","2022017","2022019","2022027","2022041","2022055","2022056","2022076","2022087","2022088","2022095","2022115","2022116","2022347","2022348","2022349","2022351","2023062","2023063","2023064","2023065","2023066","2023068","2023069","2023070","2023074","2023080","2023081","2023082","2023083","2023084","2023085"]

years = [2019, 2020, 2021, 2022, 2023]
for year in years:
    source_folder= f'fslippe@login2.nird.sigma2.no:/nird/projects/NS9600K/data/modis/cao/MOD02_npy/{year}/'
    destination_folder = f'/scratch/fslippe/modis/MOD02_npy_w_cao/{year}/'

    for wildcard in wildcards:
        command = f"rsync -av --progress {source_folder}*{wildcard}* {destination_folder}"
        os.system(command)
