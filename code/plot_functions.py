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
import importlib
import functions 
from scipy.spatial import distance_matrix
importlib.reload(functions)
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

        current_lon = np.empty((n_patches_tot[i], patch_size, patch_size))
        current_lon[np.squeeze(indices[i].numpy())] = all_lon_patches[i]
        lon_map = np.reshape(current_lon, (reduced_height, reduced_width, patch_size, patch_size))

        current_lat = np.empty((n_patches_tot[i], patch_size, patch_size))
        current_lat[np.squeeze(indices[i].numpy())] = all_lat_patches[i]
        lat_map = np.reshape(current_lat, (reduced_height, reduced_width, patch_size, patch_size))

        # Get label map

        label_map = generate_map_from_labels(labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)

        
        binary_map = np.isin(label_map, desired_label)

        # Label connected components, considering diagonal connections
        """USE OF DIAGONAL CONNECTIONS"""
        structure = ndimage.generate_binary_structure(2, 2)
        labeled_map, num_features = ndimage.label(binary_map, structure=structure)
        
        """NO DIAGONAL CONNECTIONS:"""
        #labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))
      

        # Iterate through each region and check if its size exceeds the threshold
        for region_idx, region_size in enumerate(region_sizes):
            if region_size > size_threshold:
                # fig, axs = plt.subplots(1,3)
                # fig.suptitle("%s\nCAO found for threshold %s" %(dates[i], size_threshold))
                # axs[0].imshow(x[i], cmap="gray")

                # #axs[0].invert_xaxis()
                # tab20 = plt.get_cmap("tab20")

                # # Create a custom colormap with the first 14 colors
                # custom_cmap = mcolors.ListedColormap(tab20.colors[:14])
                # cb =axs[1].imshow(label_map, cmap=custom_cmap)   
                # axs[2].imshow(np.where( np.isin(label_map, desired_label), label_map, np.nan))                

                # plt.colorbar(cb)
                # plt.show()
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


def save_img_with_labels(x, lon_lats, n_patches_tot,
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
                      global_max):
    

    cmap_tab10 = plt.cm.tab10
    cmap_tab20 = plt.cm.tab20
    colors_tab20 = cmap_tab20(np.arange(cmap_tab20.N))[1::2]
    colors_tab10 = cmap_tab10(np.arange(cmap_tab10.N))
    extra_colors = colors_tab20
    black = np.array([0, 0, 0, 1])
    colors_new = np.vstack((colors_tab10, colors_tab20))[:global_max-1]
    colors_new = np.vstack((colors_new, black))

    new_cmap = mcolors.ListedColormap(colors_new)
    # This will track which dates have been counted for each grid cell
    dates_counted = {}

    s = 0
    # Run through all images 
    for i in range(len(dates)):
        # Generate lon lat maps
        height, width = shapes[i]
        reduced_height = height // patch_size
        reduced_width = width //patch_size

        current_lon = np.empty((n_patches_tot[i], patch_size, patch_size))
        current_lon[np.squeeze(indices[i].numpy())] = all_lon_patches[i]
        lon_map = np.reshape(current_lon, (reduced_height, reduced_width, patch_size, patch_size))

        current_lat = np.empty((n_patches_tot[i], patch_size, patch_size))
        current_lat[np.squeeze(indices[i].numpy())] = all_lat_patches[i]
        lat_map = np.reshape(current_lat, (reduced_height, reduced_width, patch_size, patch_size))

        # Get label map

        label_map = generate_map_from_labels(labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)

        
        binary_map = np.isin(label_map, desired_label)

        # Label connected components, considering diagonal connections
        """USE OF DIAGONAL CONNECTIONS"""
        # structure = ndimage.generate_binary_structure(2, 2)
        # labeled_map, num_features = ndimage.label(binary_map, structure=structure)
        
        """NO DIAGONAL CONNECTIONS:"""
        labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))
      
        # Iterate through each region and check if its size exceeds the threshold
        for region_idx, region_size in enumerate(region_sizes):
            if region_size > size_threshold:
                # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10), dpi=80)
                # ax.set_extent([-40, 40, 55, 85], crs=ccrs.PlateCarree())  # Adjust depending on your lat/lon bounds
                # fig.suptitle("%s\n%s CAO found for threshold %s" %(dates[i], region_size, size_threshold))

                # # Scatter plot of points
                # ax.pcolormesh(lon_lats[i][0,:,:], lon_lats[i][1,:,:], x[i][:,:,0], transform=ccrs.PlateCarree(), cmap='gray', vmin=0, vmax=7)
                # ax.coastlines()
                # ax.gridlines()


                fig, axs = plt.subplots(1,3)
                axs[0].imshow(x[i], cmap="gray")
                fig.suptitle("%s idx: %s\n%s CAO found for threshold %s" %(dates[i], i, region_size, size_threshold))

                #axs[0].invert_xaxis()
                #tab20 = plt.get_cmap("tab20")

                # Create a custom colormap with the first 14 colors
                #custom_cmap = mcolors.ListedColormap(tab20.colors[:14])
                cb =axs[1].imshow(label_map, cmap=new_cmap)   
                axs[2].imshow(np.where( np.isin(label_map, desired_label), label_map, np.nan))                

                plt.colorbar(cb)
                plt.show()
        


def plot_img_cluster_mask(x, labels, masks, starts, ends, shapes, indices, dates, n_patches_tot, patch_size, global_min, global_max, index_list, chosen_label=2, save=None):
    # Add black to the end of cmap
    norm_mask = Normalize(vmin=0, vmax=1)  
   
    cmap_tab10 = plt.cm.tab10
    cmap_tab20 = plt.cm.tab20
    colors_tab20 = cmap_tab20(np.arange(cmap_tab20.N))[1::2]
    colors_tab10 = cmap_tab10(np.arange(cmap_tab10.N))
    extra_colors = colors_tab20
    black = np.array([0, 0, 0, 1])
    colors_new = np.vstack((colors_tab10, colors_tab20))[:global_max-1]
    colors_new = np.vstack((colors_new, black))

    new_cmap = mcolors.ListedColormap(colors_new)
    print(x[0].shape)
    
    n_bands = x[0].shape[2]
    max_bands = np.max(np.array([np.max(xi, axis=(0,1)) for xi in x]), axis=0)
    if len(max_bands) >= 3:
        max_bands[3] = 0.3
        
    print(max_bands.shape)
    if labels[0].ndim == 1:
        n_labels = len(labels)

    else:
        n_labels = 1
        labels = [labels]
        global_min = [global_min]  
        global_max = [global_max]  


    # Run through index_list corresponding to picture i
    for i in index_list:
        # Get cluster map i

        # Plot each map        
        fig, axs = plt.subplots(1, 1 + n_bands + n_labels, figsize=(15+ 5*(n_bands + n_labels) , 6))
        for j in range(n_bands):
            cb = axs[j].imshow(x[i][:,:,j], cmap="gray", vmin=0, vmax=max_bands[j])
            plt.colorbar(cb)
        for k in range(n_labels):
            norm = Normalize(vmin=global_min[k], vmax=global_max[k])
            map = generate_map_from_labels(labels[k], starts[i], ends[i], shapes[i], indices[i], global_max[k], n_patches_tot[i], patch_size)
            cb = axs[n_bands+k].imshow(map, cmap=new_cmap, norm=norm)
            plt.colorbar(cb)
        fig.suptitle("idx:%s,  dates:%s,  max:%s,  min:%s,  mean:%s,  n_lab:%s" %(i, dates[i], np.max(map), np.min(map), np.mean(map), np.sum((map.ravel()==chosen_label))))
        axs[-1].imshow(masks[i], norm=norm_mask)
        plt.tight_layout()
    
    if save != None:
        plt.savefig(save, dpi=200)

    plt.show()





from scipy.spatial import distance_matrix

def plot_map_with_nearest_neighbors(original_map, lons, lats, lon_map, lat_map):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10), dpi=200)
    ax.set_extent([-40, 40, 55, 85], crs=ccrs.PlateCarree())  # Adjust depending on your lat/lon bounds

    # Scatter plot of points
    ax.pcolormesh(lon_map, lat_map, original_map, transform=ccrs.PlateCarree(), cmap='gray')
    ax.scatter(lons, lats, color='red', s=0.1, transform=ccrs.PlateCarree())

    # distances = distance_matrix(np.column_stack([lons, lats]), np.column_stack([lons, lats]))

    # # Replace diagonal (distance to self) with a high value
    # np.fill_diagonal(distances, np.inf)

    # # For each point, find the index of the nearest point
    # nearest_indices = np.argmin(distances, axis=1)

    # for i, ni in enumerate(nearest_indices):
    #     ax.plot([lons[i], lons[ni]], [lats[i], lats[ni]], color='red', linewidth=0.5, transform=ccrs.PlateCarree())

    ax.coastlines()
    ax.gridlines()

    plt.show()


# def plot_map_with_boundaries_in_projection(original_map, lons, lats, lon_map, lat_map):
#     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10,10), dpi=200)
#     ax.set_extent([-40, 40, 55, 85], crs=ccrs.PlateCarree())  # Adjust depending on your lat/lon bounds
    
#     # Displaying the map
#     ax.pcolormesh(lon_map, lat_map, original_map, transform=ccrs.PlateCarree(), cmap='gray')
    
#     distances = distance_matrix(np.column_stack([lons, lats]), np.column_stack([lons, lats]))
    
#     # Replace diagonal (distance to self) with a high value
#     np.fill_diagonal(distances, np.inf)
    
#     # For each point, find the index of the nearest point (or two nearest points)
#     for i in range(len(lons)):
#         nearest_indices = np.argsort(distances[i])[:2]
#         for ni in nearest_indices:
#             ax.plot([lons[i], lons[ni]], [lats[i], lats[ni]], color='red', linewidth=0.5, transform=ccrs.PlateCarree())
    
#     ax.coastlines()
#     ax.gridlines()

#     plt.show()