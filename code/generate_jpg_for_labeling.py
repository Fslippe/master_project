import matplotlib.pyplot as plt
from PIL import Image


### EXTRACT CAO AND NOn CAO CASES
import importlib
import extract_training_data
importlib.reload(extract_training_data)
from extract_training_data import *
patch_size = 128


dates_block = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_block.npy")
times_block = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_block.npy")
dates_rest = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_rest.npy")
times_rest = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_rest.npy")
dates = np.append(dates_block, dates_rest)
times = np.append(times_block, times_rest)

print(dates)
bands = [29]
#folder = "/scratch/fslippe/modis/MOD02/labeling_session"
folder = "/scratch/fslippe/modis/MOD02/2019/ /scratch/fslippe/modis/MOD02/2020/ /scratch/fslippe/modis/MOD02/2021/ /scratch/fslippe/modis/MOD02/2022/ /scratch/fslippe/modis/MOD02/2023/"

#folder = "/scratch/fslippe/modis/MOD02/2020/"
x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = extract_1km_data(folder,
                                                         bands=bands,
                                                         #start_date=start_converted,
                                                         #end_date=end_converted,
                                                         date_list=dates[10],
                                                         return_lon_lat=True,
                                                         data_loc=data_loc,
                                                         data_type="npy",
                                                         combine_pics=True)

x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
x_cao = list(x_cao)
dates_cao = list(dates_cao)
len(x_cao)


for i in range(len(x_cao)):
    if (dates_cao[i], mod_min_cao[i]) in zip(dates_block, times_block):
        # Scale data to range 0-255 and convert type
        img_data = np.array((x_cao[i][:,:,0] - np.min(x_cao[i])) / (np.max(x_cao[i]) - np.min(x_cao[i])) * 255, dtype=np.uint8)
        
        # Invert the image data
        img_data = 255 - img_data
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(img_data)

        # Save the image
        image.save("/uio/hume/student-u37/fslippe/data/figures_for_labeling/block/MOD021KM.A%s.%s_block.jpg" %(dates_cao[i], mod_min_cao[i]))

for i in range(len(x_cao)):
    if (dates_cao[i], mod_min_cao[i]) in zip(dates_rest, times_rest):
        # Scale data to range 0-255 and convert type
        img_data = np.array((x_cao[i][:,:,0] - np.min(x_cao[i])) / (np.max(x_cao[i]) - np.min(x_cao[i])) * 255, dtype=np.uint8)
        
        # Invert the image data
        img_data = 255 - img_data

        # Convert numpy array to PIL Image
        image = Image.fromarray(img_data)

        # Save the image
        image.save("/uio/hume/student-u37/fslippe/data/figures_for_labeling/rest/MOD021KM.A%s.%s_rest.jpg" %(dates_cao[i], mod_min_cao[i]))

dpi = 80  # change this to match your needs


for i in range(len(x_cao)):
    if (dates_cao[i], mod_min_cao[i]) in zip(dates_block, times_block): 
        height_pixels, width_pixels = x_cao[i].shape[:2]
        height_inches = height_pixels / dpi
        width_inches = width_pixels / dpi
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        plt.imshow(x_cao[i], cmap="gray_r")
        plt.axis('off')  # to hide the axes
        plt.savefig("/uio/hume/student-u37/fslippe/data/figures_for_labeling/MOD021KM.A%s.%s_block.jpg" %(dates_cao[i], mod_min_cao[i]), dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

from PIL import Image

# Generate a saved image for testing (You should replace this as your actual image pathname)
image_path = "/uio/hume/student-u37/fslippe/data/figures_for_labeling/block/MOD021KM.A2019318.1155_block.jpg"

# Open the saved image and get dimensions
with Image.open(image_path) as img:
    width, height = img.size
    print(width, height)


import datetime

# assuming dates and times are lists of strings
for i in range(len(dates)):
    if not (dates[i], times[i]) in zip(dates_cao, mod_min_cao):
        print("%s-%s %02d:%02d" % (dates[i][:4], dates[i][4:], times[i]//100, times[i]%100))
    else:
        print(dates[i])


