import numpy as np
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from scipy import ndimage
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

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

    desired_label = 2
    size_threshold = 10
    all_region_lon = []
    all_region_lat = []
    s = 0
    for i in range(len(dates)):
        height, width = shapes[i]

        # Calculate the dimensions of the reduced resolution array
        reduced_height = height // patch_size
        reduced_width = width //patch_size
        
        current_labels = np.ones((n_patches_tot[i]))*(global_max+1)
        current_labels[np.squeeze(indices[i].numpy())] = labels[starts[i]:ends[i]]

        current_lon = np.empty((n_patches_tot[i], 64, 64))
        current_lon[np.squeeze(indices[i].numpy())] = all_lon_patches[i]
        lon_map = np.reshape(current_lon, (reduced_height, reduced_width, 64, 64))

        current_lat = np.empty((n_patches_tot[i], 64, 64))
        current_lat[np.squeeze(indices[i].numpy())] = all_lat_patches[i]
        lat_map = np.reshape(current_lat, (reduced_height, reduced_width, 64, 64))

        label_map = np.reshape(current_labels, (reduced_height, reduced_width))
        
        binary_map = (label_map == desired_label)

        # Label connected components
        labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Get the date associated with the current x[i]
        current_date = dates[i]

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
                    if current_date not in dates_counted[idx]:
                        counts.ravel()[idx] += 1
                        dates_counted[idx].add(current_date)

    return x_grid, y_grid, counts



def plot_img_cluster_mask(x, labels, masks, starts, ends, shapes, indices, dates, n_patches_tot, patch_size, global_min, global_max, index_list, save=None):
    cluster_map = []
    norm = Normalize(vmin=global_min, vmax=global_max)  
    
    norm_mask = Normalize(vmin=0, vmax=1)  

    cmap_tab10 = plt.cm.tab10
    colors_tab10 = cmap_tab10(np.arange(cmap_tab10.N))

    # Add black to the end
    black = np.array([0, 0, 0, 1])
    colors_new = np.vstack((colors_tab10, black))

    # Create a new colormap from the combined list of colors
    new_cmap = mcolors.ListedColormap(colors_new)

    for i in index_list:#range(start, start + tot_pics):
        height, width = shapes[i]

        # Calculate the dimensions of the reduced resolution array
        reduced_height = height // patch_size
        reduced_width = width //patch_size
        current_labels = np.ones((n_patches_tot[i]))*(global_max+1)
        print(labels[starts[i]:ends[i]].shape)

        current_labels[np.squeeze(indices[i].numpy())] = labels[starts[i]:ends[i]]
    
        cluster_map.append(np.reshape(current_labels, (reduced_height, reduced_width)))
        fig, axs = plt.subplots(1,3, figsize=(25, 6))
        fig.suptitle("idx:%s, dates:%s   max:%s,    min:%s,    mean:%s,    n_lab:%s" %(i, dates[i], np.max(current_labels), np.min(current_labels), np.mean(current_labels), np.sum((cluster_map[i].ravel()==2))))
        cb = axs[0].imshow(x[i], cmap="gray")
        plt.colorbar(cb)
        #axs[0].contourf(lon_lats[i][0][0,:], lon_lats[i][1][:,0], x[i][:,:,0], cmap="gray")

        #axs[0].pcolormesh(lon_lats[i][0], lon_lats[i][1], x[i][:,:,0], cmap="gray")
        cb = axs[1].imshow(cluster_map[i], cmap=new_cmap, norm=norm)
        plt.colorbar(cb)
        axs[2].imshow(masks[i], norm=norm_mask)
    plt.tight_layout()
    
    if save != None:
        plt.savefig(save, dpi=200)

    plt.show()
