import numpy as np
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from scipy import ndimage
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
from functions import *

# Define the grid in projected coordinates
def generate_hist_map(n_patches_tot,
                      indices,
                      labels,
                      starts,
                      ends,  
                      shapes,
                      all_lon_patches,
                      all_lat_patches,  
                      dates,
                      desired_label,
                      size_threshold,
                      patch_size,
                      global_max,
                      projection = ccrs.Stereographic(central_latitude=90),
                      grid_resolution = 100e3):
    
    # Generate grid to add counts on
    x_extent = [-4e6, 4e6]
    y_extent = [-4e6, 4e6]
    x_grid, y_grid = np.meshgrid(np.arange(x_extent[0], x_extent[1], grid_resolution),
                                np.arange(y_extent[0], y_extent[1], grid_resolution))

    # Initialize the count matrix
    counts = np.zeros_like(x_grid)

    # Create a KDTree for faster nearest neighbor search
    tree = cKDTree(list(zip(x_grid.ravel(), y_grid.ravel())))

    # This will track which dates have been counted for each grid cell
    dates_counted = {}

    s = 0
    # Run through all images 
    for i in range(len(dates)):
        # Generate lon lat maps
        height, width = shapes[i]
        reduced_height = height // patch_size
        reduced_width = width //patch_size

        current_lon = np.empty((n_patches_tot[i], 64, 64))
        current_lon[np.squeeze(indices[i].numpy())] = all_lon_patches[i]
        lon_map = np.reshape(current_lon, (reduced_height, reduced_width, 64, 64))

        current_lat = np.empty((n_patches_tot[i], 64, 64))
        current_lat[np.squeeze(indices[i].numpy())] = all_lat_patches[i]
        lat_map = np.reshape(current_lat, (reduced_height, reduced_width, 64, 64))

        # Get label map
        label_map = generate_map_from_labels(labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)
        binary_map = (label_map == desired_label)
        
        # Label connected components
        labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))


        # Iterate through each region and check if its size exceeds the threshold
        for region_idx, region_size in enumerate(region_sizes):
            if region_size > size_threshold:
                # Get the indices of the region
                region_coordinates = np.where(labeled_map == region_idx)
                
                # Convert to projected coordinates
                x_proj, y_proj = projection.transform_points(ccrs.PlateCarree(), 
                                                            lon_map[region_coordinates].ravel(), 
                                                            lat_map[region_coordinates].ravel())[:, :2].T
                s+=1

                # Query the KDTree for nearest grid points
                _, idxs = tree.query(list(zip(x_proj, y_proj)))  

                # Check and Increment the counts based on date condition
                for idx in idxs:
                    if idx not in dates_counted:
                        dates_counted[idx] = set()
                    if dates[i] not in dates_counted[idx]:
                        counts.ravel()[idx] += 1
                        dates_counted[idx].add(dates[i])

    return x_grid, y_grid, counts





def plot_img_cluster_mask(x, labels, masks, starts, ends, shapes, indices, dates, n_patches_tot, patch_size, global_min, global_max, index_list, chosen_label=2, save=None):
    # Add black to the end of cmap
    norm = Normalize(vmin=global_min, vmax=global_max)  
    norm_mask = Normalize(vmin=0, vmax=1)  
    cmap_tab10 = plt.cm.tab20
    colors_tab10 = cmap_tab10(np.arange(cmap_tab10.N))
    black = np.array([0, 0, 0, 1])
    colors_new = np.vstack((colors_tab10, black))
    new_cmap = mcolors.ListedColormap(colors_new)
    print(x[0].shape)
    
    n_bands = x[0].shape[2]
    max_bands = np.max(np.array([np.max(xi, axis=(0,1)) for xi in x]), axis=0)
    max_bands[3] = 0.3
    print(max_bands.shape)
    # Run through index_list corresponding to picture i
    for i in index_list:
        # Get cluster map i
        map = generate_map_from_labels(labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)

        # Plot each map        
        fig, axs = plt.subplots(1,2+ n_bands, figsize=(20+ 5*n_bands , 6))
        fig.suptitle("idx:%s,  dates:%s,  max:%s,  min:%s,  mean:%s,  n_lab:%s" %(i, dates[i], np.max(map), np.min(map), np.mean(map), np.sum((map.ravel()==chosen_label))))
        for j in range(n_bands):
            cb = axs[j].imshow(x[i][:,:,j], cmap="gray", vmin=0, vmax=max_bands[j])
            plt.colorbar(cb)
        cb = axs[-2].imshow(map, cmap=new_cmap, norm=norm)
        plt.colorbar(cb)
        axs[-1].imshow(masks[i], norm=norm_mask)
        plt.tight_layout()
    
    if save != None:
        plt.savefig(save, dpi=200)

    plt.show()
