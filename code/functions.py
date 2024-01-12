import numpy as np 
from scipy import ndimage
from scipy.stats import linregress
import cartopy.crs as ccrs
import pyproj
from scipy.spatial import cKDTree
from autoencoder import * 
import datetime
import xarray as xr
import pyproj
from shapely.geometry import Point, Polygon
from plot_functions import * 
geodesic = pyproj.Geod(ellps='WGS84')


def process_label_maps(labels, all_lon_patches, all_lat_patches, starts_cao, ends_cao, shapes_cao, indices_cao, global_max, n_patches_tot_cao, patch_size, strides, label_1, label_2, size_thr_1=20, size_thr_2=20):
    def calculate_patch_mean(patches):
        if patches.ndim == 2:
            return np.mean(np.expand_dims(patches, axis=0), axis=(1, 2))
        else:
            return np.mean(patches, axis=(1, 2))
   
    pat_lon = [calculate_patch_mean(patch) for patch in all_lon_patches]
    pat_lat = [calculate_patch_mean(patch) for patch in all_lat_patches]

    pat_lon = np.concatenate(np.array(pat_lon), axis=0)
    pat_lat = np.concatenate(np.array(pat_lat), axis=0)
    
    label_map = np.empty(len(starts_cao), dtype=object)
    lon_map = np.empty(len(starts_cao), dtype=object)
    lat_map = np.empty(len(starts_cao), dtype=object)
    
    index_list = range(len(starts_cao))
    
    for i in index_list:
        label_map[i] = generate_map_from_labels(labels, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], global_max, n_patches_tot_cao[i], patch_size, strides)
        label_map[i] = remove_labels_from_size_thresholds(label_map[i], label_1, label_2, size_thr_1=size_thr_1, size_thr_2=size_thr_2)
        lon_map[i] = generate_map_from_labels(pat_lon, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], global_max, n_patches_tot_cao[i], patch_size, strides)
        lat_map[i] = generate_map_from_labels(pat_lat, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], global_max, n_patches_tot_cao[i], patch_size, strides)

    return label_map, lon_map, lat_map

def calculate_scores_and_plot(model_boundaries, model_areas, labeled_boundaries, labeled_areas, plot=False):
    area_scores = []  # To store the area and border scores
    border_scores = []  # To store the area and border scores
    weighted_area_score
    max_boundary = np.max(labeled_boundaries)

    for (m_border, m_area, l_border, l_area) in zip(model_boundaries, model_areas, labeled_boundaries, labeled_areas):
        area_diff = np.abs(m_area - l_area)
        area_score = 1 - np.nanmean(area_diff) 
        agreement = np.abs(l_area - 0.5)*10
        weighted_area_score.append(np.nanmean(area_diff * l_area))
        border_score = 1 - np.nanmean(np.abs(m_border - l_border)) 
        area_scores.append(area_score)
        border_scores.append(area_score)


        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(m_area)
            axs[1].imshow(l_area)
            plt.show()

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(m_border)
            axs[1].imshow(l_border)
            plt.show()

            fig, axs = plt.subplots(1, 2)
            cb1 = axs[0].imshow(np.abs(m_area - l_area))
            plt.colorbar(cb1, ax=axs[0])
            cb2 = axs[1].imshow(np.abs(m_border - l_border))
            plt.colorbar(cb2, ax=axs[1])
            plt.show()

    weighted_border_score = [b * max_boundary for b in border_scores]
    
    return area_scores, border_scores, weighted_area_score, weighted_border_score


def process_model_masks(index_list, lon_map, lat_map, valid_lons, valid_lats, indices_cao, label_map, label_1, label_2, plot=False):
    brush = gaussian_brush(width=5, height=5, sigma=1.2, strength=1)
    model_boundaries = []
    model_areas = []
    
    for i in index_list:
        boundary_mask = np.zeros_like(lon_map[i], dtype=np.float)
        closest_indices = find_closest_indices(lon_map[i], lat_map[i], valid_lons, valid_lats)

        for (x, y) in closest_indices:
            apply_brush(boundary_mask, y, x, brush)

        valid_pos = indices_cao[i].numpy()  # This should be a flattened list or 1D np.ndarray of valid indices
        valid_mask = np.full(boundary_mask.shape, False)  # Start with a mask of False (invalid) values
        valid_mask.flat[valid_pos] = True  # Set positions defined by valid_pos to True (valid)
        boundary_mask[~valid_mask] = np.nan  # Set invalid positions in boundary_mask to np.nan

        area_mask = np.where((label_map[i] == label_1) | (label_map[i] == label_2), 1, 0).astype(np.float)
        area_mask[~valid_mask] = np.nan

        model_boundaries.append(boundary_mask)
        model_areas.append(area_mask)

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=[10, 10])
            axs[0].imshow(area_mask)
            axs[1].imshow(boundary_mask)
            plt.show()

    return model_boundaries, model_areas

def get_valid_lons_lats(x_i, lon_lats_cao, label_map, lon_map, lat_map, date, time, open_label, closed_label, p_level=950, angle_thr=5, size_threshold_1=None, size_threshold_2=None, plot=False, extent= [-15, 25, 58, 84]):
    print(date, time)
    lons, lats, angles = compute_boundary_coordinates_between_labels_2(label_map, lon_map, lat_map, open_label, closed_label, size_threshold_1=size_threshold_1, size_threshold_2=size_threshold_2)
    lons_full = lons
    lats_full = lats

    valid_lons = []
    valid_lats = []
    threshold = 90

    for lon, lat, angle in zip(lons, lats, angles):
        wind_dir = find_wind_dir_at_ll_time(lon, lat, p_level, date, time) 

        check = check_angle_threshold(wind_dir, lons, lats, lon, lat, angle_thr, min_distance=0)
        #lons, lats = check_angle_threshold_downwind(wind_dir, lons, lats, lon, lat, 5, min_distance=100000)
        if not check:
            if np.min([abs(angle - wind_dir), abs(angle - wind_dir - 360), abs(angle - wind_dir + 360)]) < threshold:
                valid_lons.append(lon)# if angle==237.43814048068378 else 0)
                valid_lats.append(lat)# if angle==237.43814048068378 else 70)
                #valid_angles.append(angle)
   
    if plot:
        ax = plot_map_with_nearest_neighbors(x_i, valid_lons, valid_lats, lon_lats_cao[0], lon_lats_cao[1], extent, figsize=(14, 10))
        # plt.quiver(valid_lons, valid_lats, valid_angles, 300)
        plt.show()
    
    return valid_lons, valid_lats
    
def find_closest_indices(grid_lons, grid_lats, lons_list, lats_list):
    index_list = []

    for lon, lat in zip(lons_list, lats_list):
        min_distance = None
        closest_index = None

        for i in range(grid_lons.shape[0]):
            for j in range(grid_lons.shape[1]):
                _, _, distance = geodesic.inv(lon, lat, grid_lons[i,j], grid_lats[i,j])
                
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_index = (i, j)

        index_list.append(closest_index)

    return index_list


def get_area_mask(boundary_coordinates, mask_shape):
    polygon = Polygon(boundary_coordinates)
    minx, miny, maxx, maxy = polygon.bounds
    points_inside = []

    # Loop through the grid of points covering the bounding box
    for x in range(int(minx), int(maxx) + 1):
        for y in range(int(miny), int(maxy) + 1):
            point = Point(x, y)

            # Check if the current point is inside the polygon
            if polygon.contains(point):
                points_inside.append((x, y))
    
    # Create a mask of the specified shape and set points inside the polygon to True (or 1)
    mask = np.full(mask_shape, False)  # or np.zeros(mask_shape, dtype=bool)
    for x, y in points_inside:
        # Assure that point indices are within bounds of mask
        if 0 <= x < mask_shape[1] and 0 <= y < mask_shape[0]:
            mask[y, x] = True  # or 1 for a binary mask

    return mask

    

def get_area_and_border_mask(x_cao, dates, times, masks_cao, df, dates_cao, mod_min_cao, reduction, patch_size=128, plot=False):
    downscaled_areas = []
    downscaled_borders = []
    for d, t in zip(dates, times):
        extracted_rows = df[df["image_id"].str.split("/").str[1].str.split("_").str[0] == f"MOD021KM.A{d}.{t}"]
        if len(extracted_rows) > 6:
            if plot:
                fig, axs = plt.subplots(1,3, figsize=[35, 20])

            interpolated_border = []
            interpolated_area_i = []
            for i in range(len(extracted_rows)):
                interpolated_area = []
                interpolated_area_mask = []
                di = extracted_rows.iloc[i]
                date_img = str(di["image_id"].split("/")[1].split(".")[1][1:])
                time_img = int(di["image_id"].split("/")[1].split(".")[2].split("_")[0])
                idx = np.where((np.array(dates_cao) == date_img) & (np.array(mod_min_cao) == time_img))[0][0]

                area_lines = np.array(di["data.areaLines"])
                border_lines = np.array(di["data.borderLines"], dtype=object)
                reduced_height = (x_cao[idx].shape[0] - patch_size) // reduction + 1
                reduced_width = (x_cao[idx].shape[1] - patch_size) // reduction + 1
                scale_factor_y = reduced_height / x_cao[idx].shape[0]
                scale_factor_x = reduced_width / x_cao[idx].shape[1]


                n_areas = len(area_lines)
                if n_areas > 0:
                    for j in range(n_areas):
                        area = np.array(area_lines[j])
                        interpolated_area_boundary = interpolate_coords(area, connect_first_last=True)
                        scaled_boundary_coordinates = np.copy(interpolated_area_boundary.astype(float))
                        scaled_boundary_coordinates[:, 0] *= scale_factor_x
                        scaled_boundary_coordinates[:, 1] *= scale_factor_y
                        area_mask = get_area_mask(scaled_boundary_coordinates, (reduced_height, reduced_width))

                        #area_mask = get_area_mask(interpolated_area_boundary // reduction, (x_cao[idx].shape[0] // reduction, x_cao[idx].shape[1] // reduction))

                        interpolated_area.append(interpolated_area_boundary)
                        interpolated_area_mask.append(area_mask)

                    interpolated_sum = np.sum(interpolated_area_mask, axis=0)
                    interpolated_area_i.append(np.where(interpolated_sum > 1, 1, interpolated_sum))

                if plot:
                    axs[0].imshow(x_cao[idx], cmap="gray_r")
                    for k in range(len(interpolated_area)):
                        #axs[0].scatter(interpolated_area[k].T[0] // reduction, interpolated_area[k].T[1] // reduction, s=0.05, color="r")
                        #axs[0].imshow(interpolated_area[k], alpha=0.8/len(extracted_rows), cmap="Reds")
                        axs[0].fill(interpolated_area[k].T[0], interpolated_area[k].T[1], alpha=0.8/len(extracted_rows), color="r")

                n_borders = len(border_lines)
                if n_borders > 0:
                    for j in range(n_borders):
                        border = np.array(border_lines[j])
                        interpolated_border.append(interpolate_coords(border.astype(float), connect_first_last=False))
            
                
                idx = np.where((np.array(dates_cao) == date_img) & (np.array(mod_min_cao) == time_img))[0][0]
                if plot:
                    axs[1].imshow(x_cao[idx], cmap="gray_r")
                    for k in range(len(interpolated_border)):
                        axs[1].scatter(interpolated_border[k].T[0], interpolated_border[k].T[1], s=0.5, color="r")

            # Final areas
            interpolated_sum_i = np.sum(interpolated_area_i, axis=0) / len(extracted_rows)
            downscaled_areas.append(interpolated_sum_i)

            

            # Generate the Gaussian brush (adjust width, height, and sigma as needed)
            brush = gaussian_brush(width=65, height=65, sigma=16, strength=1/len(extracted_rows))

            tot_border = np.zeros(x_cao[idx].shape[:2])
            tot_border_reduced = np.zeros((reduced_height, reduced_width))
            
            # Iterate through your border coordinates and apply the brush
            for border_coords in interpolated_border:
                border_mask = np.zeros(x_cao[idx].shape[:2])
                for x, y in border_coords:
                    apply_brush(border_mask, int(x), int(y), brush)
                tot_border += border_mask 

            for i in range(reduced_height):
                for j in range(reduced_width):
                    tot_border_reduced[i, j] = np.mean(tot_border[i * reduction: (i + 1) * reduction, j * reduction: (j + 1) * reduction])
            downscaled_borders.append(tot_border_reduced)

            if plot:
                cb = axs[2].imshow(tot_border_reduced, vmin=0, vmax=1)
                plt.colorbar(cb)
                plt.show()

        return downscaled_areas, downscaled_borders


def gaussian_brush(width=5, height=5, sigma=1.0, strength=1):
    """
    Create a 2D Gaussian brush centered in the middle of the width and height.
    """
    x, y = np.meshgrid(np.linspace(-width/2, width/2, width),
                       np.linspace(-height/2, height/2, height))
    d = np.sqrt(x*x + y*y)
    g = strength*np.exp(-(d**2 / (2.0 * sigma**2)))
    return g


def apply_brush(mask, x, y, brush):
    """
    Apply the given brush to the mask at position x, y.
    """
    half_width = brush.shape[1] // 2
    half_height = brush.shape[0] // 2

    col_start = max(0, x - half_width)
    col_end = col_start + brush.shape[1]

    row_start = max(0, y - half_height)
    row_end = row_start + brush.shape[0]

    if col_end > mask.shape[1]:
        col_end = mask.shape[1]
        col_start = col_end - brush.shape[1]

    if row_end > mask.shape[0]:
        row_end = mask.shape[0]
        row_start = row_end - brush.shape[0]

    mask[row_start:row_end, col_start:col_end] = np.where(
        mask[row_start:row_end, col_start:col_end] < brush, brush, mask[row_start:row_end, col_start:col_end])
    #mask[row_start:row_end, col_start:col_end] = mask[row_start:row_end, col_start:col_end] + brush
   # mask[row_start:row_end, col_start:col_end] = brush

    return mask


def bresenham_line(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to generate points between start and end."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def interpolate_coords(coords, connect_first_last):
    """Interpolate between points in coords if they are not neighbors."""
    interpolated = []
    for i in range(len(coords) - 1):
        start = coords[i] 
        end = coords[i + 1]
        if start[0] < 0:
            start[0] = 0 
        if start[1] < 0:
            start[1] = 0
        if end[0] < 0:
            end[0] = 0 
        if end[1] < 0:
            end[1] = 0            

        # Check if points are neighbors
        if max(abs(round(start[0]) - round(end[0])), abs(round(start[1]) - round(end[1]))) > 1:
            interpolated.extend(bresenham_line(
                round((start[0])), round((start[1])), round((end[0])), round((end[1]))))
        else:
            interpolated.append((round(start[0]), round(start[1])))

    interpolated.append(coords[-1])  # Add the last point

    if connect_first_last:
        start = (coords[-1])
        end = (coords[0])
        # Check if points are neighbors
        if max(abs(round(start[0]) - round(end[0])), abs(round(start[1]) - round(end[1]))) > 1:
            interpolated.extend(bresenham_line(
                round((start[0])), round((start[1])), round((end[0])), round((end[1]))))
        else:            
            interpolated.append((round(start[0]), round(start[1])))

    
    return np.array(interpolated)  



def generate_xy_grid(x_extent = [-2.2e6, 2.2e6], y_extent = [-3.6e6, -0.5e6], grid_resolution=128e3):
    x_grid, y_grid = np.meshgrid(np.arange(x_extent[0], x_extent[1], grid_resolution),
                                np.arange(y_extent[0], y_extent[1], grid_resolution))
    return x_grid, y_grid
    
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
                      grid_resolution = 128e3):
    
    # Generate grid to add counts on
   
    x_grid, y_grid = generate_xy_grid(grid_resolution=grid_resolution)
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
            if region_size >= size_threshold:
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


def find_wind_dir_at_ll_time(lon, lat, p_level, date, time):
    datetime_obj = datetime.datetime.strptime("%s%s" %(date, time), "%Y%j%H%M")
    formatted_date = datetime_obj.strftime("%Y%m%d")
    year = str(date)[:4]
    ds = xr.open_dataset("/scratch/fslippe/MERRA/%s/MERRA2_400.inst3_3d_asm_Np.%s.SUB.nc" % (year, formatted_date))
    ds_time = ds.sel(time=datetime_obj, lon=lon, lat=lat, lev=p_level, method='nearest')
    wind_dir = np.degrees(np.arctan2(ds_time['V'], ds_time['U']))
    
    wind_dir = (wind_dir +270) % 360
    return np.float32(wind_dir)


def check_angle_threshold(wind_direction, lons, lats, lon, lat, threshold, min_distance):
    new_lon, new_lat = step_against_wind(lon, lat, (wind_direction+180) % 360, step_distance = 100)
    for lon_i, lat_i in zip(lons, lats):
        # Calculate vector angle for the current pair of longitude and latitude

        _, angle, distance = geodesic.inv(new_lon, new_lat, lon_i, lat_i)
        if distance > min_distance:
            vector_angle = (angle + 360) % 360
            # Check if the angle difference is within the threshold range of the wind direction
            if abs((vector_angle - wind_direction + 180) % 360 - 180) <= threshold:
                return True  # Return True if any angle is within the threshold

    return False # Return False if no angles are within the threshold


def check_angle_threshold_downwind(wind_direction, lons, lats, lon, lat, threshold, min_distance):
    filtered_lons = []
    filtered_lats = []
    idx = []
    for i, (lon_i, lat_i) in enumerate(zip(lons, lats)):
        # Calculate vector angle for the current pair of longitude and latitude
        _, angle, distance = geodesic.inv(lon, lat, lon_i, lat_i)
        if distance > min_distance:
            vector_angle = (angle + 360) % 360
            # Check if the angle difference is within the threshold range of the wind direction
            if not abs((vector_angle - wind_direction) % 360 - 180) <= threshold:
                filtered_lons.append(lon_i)
                filtered_lats.append(lat_i)


        else:
            filtered_lons.append(lon_i)
            filtered_lats.append(lat_i)

    return filtered_lons, filtered_lats  

def step_against_wind(lon, lat, wind_direction, step_distance = 32):
    step_lon = step_distance / (111.111 * np.cos(np.radians(lat)))

    new_lon = lon + step_lon * np.sin(np.radians(wind_direction-180))
    new_lat = lat + step_distance / 111.111 * np.cos(np.radians(wind_direction))
    return new_lon, new_lat

def is_closer_to_any_point(original_lon, original_lat, new_lon, new_lat, existing_lons, existing_lats):
    # Check if the new point is closer to any existing point
    original_distance = geodesic.inv(new_lon, new_lat, original_lon, original_lat)[2]

    for (existing_lon, existing_lat) in zip(existing_lons, existing_lats):
        distance = geodesic.inv(new_lon, new_lat, existing_lon, existing_lat)[2]
        # Extract the distance from the result
        if distance < original_distance:  # You may adjust this threshold
            return True
    return False


def convert_to_day_of_year(date_str):
    # Parse the date
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Convert to datetime object
    date_obj = datetime.datetime(year, month, day)

    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday

    # Return in the desired format
    return f"{year}{day_of_year:03d}"  # Using :03d to ensure it's a 3-digit number


def generate_date_list(start, end):
    start_date = datetime.datetime.strptime(start, '%Y%m%d')
    end_date = datetime.datetime.strptime(end, '%Y%m%d')
    
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(convert_to_day_of_year(current_date.strftime('%Y%m%d')))
        current_date += datetime.timedelta(days=1)
    return date_list

def convert_to_standard_date(date_str):
    # Parse the date
    year = int(date_str[:4])
    day_of_year = int(date_str[4:])

    # Convert to datetime.datetime object
    date_obj = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)  # Using day_of_year - 1 because datetime.timedelta is 0-indexed

    # Return in the desired format
    return date_obj.strftime('%Y%m%d')



def generate_map_from_labels(labels, start, end, shape, idx, global_max, n_patches, patch_size, stride=None):
    # Calculate the dimensions of the reduced resolution array
    height, width = shape
    
    if stride is None or stride == patch_size:
        reduced_height = height // patch_size
        reduced_width = width // patch_size 
    else:
        reduced_height = (height - patch_size) // stride + 1
        reduced_width = (width - patch_size) // stride + 1    

    # Generate an empty map with all values set to global_max + 1
    cluster_map = np.full((reduced_height, reduced_width), global_max , dtype=labels.dtype)

    # Get the indices corresponding to the patches
    patch_indices = np.squeeze(idx.numpy())

    # Ensure the provided indices are within the expected range
    valid_indices = patch_indices < n_patches
    patch_indices = patch_indices[valid_indices]

    # Set the labels for the patches with valid indices
    cluster_map.flat[patch_indices] = labels[start:end][valid_indices]

    return cluster_map



import numpy as np

def generate_map_from_patches(patches, start, end, shape, patch_size, idx):
    num_patches_y, num_patches_x = shape[0] // patch_size, shape[1] // patch_size
    reduced_height, reduced_width = num_patches_y * patch_size, num_patches_x * patch_size

    # Create an empty map of the reduced resolution
    reconstructed_image = np.zeros((reduced_height, reduced_width))

    # Extract the patches corresponding to this image
    image_patches = patches[start:end]

    for i in range(len(image_patches)):
        if i >= len(idx):
            break

        y = int(idx[i][0] // num_patches_x) * patch_size
        x = int(idx[i][0] % num_patches_x) * patch_size

        # Place the patch in the correct position
        reconstructed_image[y:y+patch_size, x:x+patch_size] = image_patches[i]

    return reconstructed_image




def reconstruct_from_patches(patches, shapes, starts, ends, patch_size):
    reconstructed_images = []
    
    for i, shape in enumerate(shapes):
        # Create an empty image of the shape
        reconstructed_image = np.zeros((shape[0], shape[1], patches[0].shape[2]))
        
        # Extract the patches corresponding to this image
        image_patches = patches[starts[i]:ends[i]]
        
        # Place each patch into the empty image
        patch_idx = 0
        for y in range(0, shape[0], patch_size):
            for x in range(0, shape[1], patch_size):
                reconstructed_image[y:y+patch_size, x:x+patch_size, :] = image_patches[patch_idx]
                patch_idx += 1
        
        # Append the reconstructed image to the list
        reconstructed_images.append(reconstructed_image)
    
    return reconstructed_images

def shuffle_in_unison(*args):
    rng_state = np.random.get_state()
    for array in args:
        np.random.set_state(rng_state)
        np.random.shuffle(array)

        
def generate_patches(x, masks, lon_lats, max_vals, min_vals, autoencoder, strides = [None, None, None, None], lon_lat_min_max=[-35, 45, 60, 82]):
    all_patches = []
    all_lon_patches = []
    all_lat_patches = []

    starts = []
    ends =[]
    shapes = []
    start = 0 
    n_patches_tot = [] 
    indices = []


    #encoder = load_model("/uio/hume/student-u37/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
    #normalized_patches = np.concatenate([autoencoder.extract_patches(n_d) for n_d in normalized_data], axis=0)

    i=0
    tot = len(x)
    for (image, mask, lon_lat) in zip(x, masks, lon_lats):
        print(f"{i}/{tot}", end="\r")
        shapes.append(image.shape[0:2])
        patches, idx, n_patches, lon, lat = autoencoder.extract_patches(image,
                                                                            mask,
                                                                            mask_threshold=0.9,
                                                                            lon_lat=lon_lat,
                                                                            extract_lon_lat=True,
                                                                            strides=strides,
                                                                            lon_lat_min_max=lon_lat_min_max)  # Assuming this function extracts and reshapes patches for a single image
        #patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
        #n_patches = len(patches)
        

        all_patches.append(patches)
        all_lon_patches.append(lon)
        all_lat_patches.append(lat)
    

        starts.append(start)
        ends.append(start + len(patches))
        n_patches_tot.append(n_patches)
        indices.append(idx)
        start += len(patches)
        i+=1
    # Stack filtered patches from all images
    patches = (np.concatenate(all_patches, axis=0) - min_vals) / (max_vals - min_vals)

    return patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices


def get_patches_of_img_cao(labels, patches, starts, ends, shapes, indices, global_max, n_patches_tot, desired_label, size_threshold, n,  patch_size):
    """
    Find pictures with regions of patches of a desired label of sizes higher than given threshold 
    """
    patches_w = []

    for i in range(n):
        label_map = generate_map_from_labels(labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)
        
        binary_map = (label_map == desired_label)
        
        # Label connected components
        labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))
        # Iterate through each region and check if its size exceeds the threshold

        for region_idx, region_size in enumerate(region_sizes):
            if region_size > size_threshold:
                patches_w.append(patches[starts[i]:ends[i]])
    patches_w = np.concatenate(patches_w)
    return patches_w


import numpy as np
import pyproj
from skimage import measure
from shapely.geometry import LineString
import matplotlib.pyplot as plt



def perpendicular_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line defined by two endpoints.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    numerator = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    return numerator / denominator if denominator != 0 else 0

def douglas_peucker(point_list, epsilon):
    """
    Douglas-Peucker algorithm for line simplification.
    Returns the simplified coordinates and the indices of points inside the epsilon.
    """
    dmax = 0
    index = 0
    end = len(point_list)
    tot_indices = []
    for i in range(2, end - 1):
        d = perpendicular_distance(point_list[i], point_list[0], point_list[end - 1])
        if d > dmax:
            index = i
            dmax = d
        

    result_list = []
    indices_inside_epsilon = [False] * len(point_list)

    if dmax > epsilon:
        rec_results1, rec_indices1 = douglas_peucker(point_list[:index + 1], epsilon)
        rec_results2, rec_indices2 = douglas_peucker(point_list[index:], epsilon)

        # Build the result list
        result_list = rec_results1[:-1] + rec_results2
        indices_inside_epsilon[:index + 1] = rec_indices1
        indices_inside_epsilon[index:] = rec_indices2
    else:
        result_list = [point_list[0], point_list[end - 1]]

    return result_list, indices_inside_epsilon


from scipy.interpolate import CubicSpline

def simplify_line(coords, tolerance):
    line = LineString(coords)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    return np.array(simplified_line.xy).T

def compute_boundary_coordinates_between_labels(m, lon_map, lat_map, label1, label2, max_distance_to_avg=None, size_threshold_1=None, size_threshold_2=None, simplification_tolerance=0.1):
    lons = []
    lats = []
    angles = []
    orientations = []

    geodesic = pyproj.Geod(ellps='WGS84')

    if size_threshold_1:
        m_max = np.max(m)
        binary_map = np.isin(m, [label1])
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the threshold
        for region_label in range(1, num_features + 1): # Skipping background (label 0)
            if region_sizes[region_label] < size_threshold_1:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max
    if size_threshold_2:
        m_max = np.max(m)
        binary_map = np.isin(m, [label2])
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the threshold
        for region_label in range(1, num_features + 1): # Skipping background (label 0)
            if region_sizes[region_label] < size_threshold_2:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == label1:
                neighbors = [
                    (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                    (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
                ]


                for ni, nj in neighbors:
                    if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
                        if m[ni, nj] == label2:
                            # Interpolate lon/lat values for the boundary
                            interp_lon = (lon_map[i, j] + lon_map[ni, nj]) / 2
                            interp_lat = (lat_map[i, j] + lat_map[ni, nj]) / 2

                            # Calculate relative positions of label1 and label2
                            label_position = (lon_map[i, j] - lon_map[ni, nj], lat_map[i, j] - lat_map[ni, nj])
                            orientations.append(label_position)
                            #_, angle, _ = geodesic.inv(lon_map[i, j], lat_map[i, j], lon_map[ni, nj], lat_map[ni, nj])
                            #orientations.append((angle +360 ) % 360)

                            lons.append(interp_lon)
                            lats.append(interp_lat)


    
    # Combine lon and lat coordinates into a single array
    
    coords = np.column_stack((lons, lats))
    
    # Simplify the boundary using the Ramer–Douglas–Peucker algorithm
    simplified_coords = simplify_line(coords, simplification_tolerance)
    #simplified_coords, indices_inside = douglas_peucker(coords, simplification_tolerance)
    #print(indices_inside)
    #print(np.array(simplified_coords))
    #print(len(simplified_coords))
    #print(len(coords))
    plt.figure(figsize=[10,10])
    plt.scatter(coords[:,0], coords[:,1])
    for i in range(len(np.array(simplified_coords))):
        plt.scatter(np.array(simplified_coords)[i,0], np.array(simplified_coords)[ i,1], s=100, label="%s" %(i+1))

    plt.legend()
    # Calculate angles for the simplified coordinates
    for i in range(1, len(simplified_coords)):
        lon1, lat1 = simplified_coords[i - 1]
        lon2, lat2 = simplified_coords[i]
        plt.plot([lon1, lon2], [lat1,lat2])
        # Calculate the angle
        _, angle, _ = geodesic.inv(lon1, lat1, lon2, lat2)

        angle = (angle + 360) % 360
        
        # Find indices of points in the original coordinates that match the simplified coordinates
        indices = np.where(np.logical_and(lons == lon1, lats == lat1))
        idx1 = indices[0][0] if len(indices[0]) > 0 else None
        
        indices = np.where(np.logical_and(lons == lon2, lats == lat2))
        idx2 = indices[0][0] if len(indices[0]) > 0 else None
        
        # Fill angles for points between idx1 and idx2
        if idx1 is not None and idx2 is not None:
            linear_vec = (lon2-lon1, lat2-lat1)
            avg_orientation = np.mean(orientations[idx1:idx2], axis=0)
            # avg_orientation = mean_angle(orientations[idx1:idx2])
            cross_product = linear_vec[0] * avg_orientation[1] - linear_vec[1] * avg_orientation[0] 
            angle_right = True if cross_product < 0 else False
            for j in range(idx1, idx2):
                angles.append((angle-90) % 360 if angle_right else (angle +90) % 360)
    plt.show()


        
    print(len(angles))
    return lons, lats, angles


def mean_angle(angles):
    """
    Calculate the mean angle of a list of angles in degrees.
    """
    angles_rad = np.radians(angles)
    mean_vector = np.mean(np.exp(1j * angles_rad))
    mean_angle_rad = np.angle(mean_vector)
    mean_angle_deg = np.degrees(mean_angle_rad)
    return (mean_angle_deg + 360) % 360  # Convert back to the [0, 360) range


def calculate_angle(slope):
    return np.degrees(np.arctan(slope))

# Function to fit linear regression and calculate angle
def fit_linear_regression(coords):
    x, y = coords[:, 0], coords[:, 1]
    slope, _, _, _, _ = linregress(x, y)
    vec = (slope, 1)
    return vec





def rotate_vector(vector, angle):
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Define rotation matrix for the specified angle
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Rotate the vector using the matrix
    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector

def find_closest_angle(initial_angle, avg_orientation):
    # Convert initial_angle and avg_orientation to radians
    initial_angle_rad = np.radians(initial_angle)
    avg_orientation_rad = np.radians(avg_orientation)

    # Create unit vectors in the direction of initial_angle and avg_orientation
    initial_vector = np.array([np.cos(initial_angle_rad), np.sin(initial_angle_rad)])
    avg_orientation_vector = np.array([np.cos(avg_orientation_rad), np.sin(avg_orientation_rad)])

    # Rotate the initial vector both left and right
    rotated_left = rotate_vector(initial_vector, 90)  # Rotate left
    rotated_right = rotate_vector(initial_vector, -90)  # Rotate right

    # Calculate the angular differences between the rotated vectors and avg_orientation
    diff_left = np.abs(np.degrees(np.arccos(np.dot(rotated_left, avg_orientation_vector))))
    diff_right = np.abs(np.degrees(np.arccos(np.dot(rotated_right, avg_orientation_vector))))

    # Choose the rotation direction that gives the minimum angular difference
    if np.min(diff_left) < np.min(diff_right):
        return initial_angle - 90  # Rotate left
    else:
        return initial_angle + 90  # Rotate right


def compute_boundary_coordinates_between_labels_1(m, lon_map, lat_map, label1, label2, max_distance_to_avg=None, size_threshold_1=None, size_threshold_2=None, n_closest=10):
    lons = []
    lats = []
    angles = []
    orientations = []

    geodesic = pyproj.Geod(ellps='WGS84')

    if size_threshold_1:
        m_max = np.max(m)
        binary_map = np.isin(m, [label1])
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the threshold
        for region_label in range(1, num_features + 1): # Skipping background (label 0)
            if region_sizes[region_label] < size_threshold_1:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max

    if size_threshold_2:
        m_max = np.max(m)
        binary_map = np.isin(m, [label2])
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the threshold
        for region_label in range(1, num_features + 1): # Skipping background (label 0)
            if region_sizes[region_label] < size_threshold_2:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == label1:
                neighbors = [
                    (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                    (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
                ]


                for ni, nj in neighbors:
                    if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
                        if m[ni, nj] == label2:
                            # Interpolate lon/lat values for the boundary
                            interp_lon = (lon_map[i, j] + lon_map[ni, nj]) / 2
                            interp_lat = (lat_map[i, j] + lat_map[ni, nj]) / 2

                            # Calculate relative positions of label1 and label2
                            label_position = (lon_map[i, j] - lon_map[ni, nj], lat_map[i, j] - lat_map[ni, nj])
                            orientations.append(label_position)
                            #_, angle, _ = geodesic.inv(lon_map[i, j], lat_map[i, j], lon_map[ni, nj], lat_map[ni, nj])
                            #orientations.append((angle +360 ) % 360)

                            lons.append(interp_lon)
                            lats.append(interp_lat)


    
    # Combine lon and lat coordinates into a single array
    lons = np.array(lons)
    lats = np.array(lats)
    orientations = np.array(orientations)


    coords = np.column_stack((lons, lats))
    all_vecs = []

    for i in range(len(lons)):
        # Calculate distances to all other points
        distances = [geodesic.inv(lons[i], lats[i], lon2, lat2)[2] for lon2, lat2 in zip(lons, lats)]
        #_, _, distance = geodesic.inv(lons, lats, lons[i], lats[i])
        #distances = np.sqrt((lons - lons[i])**2 + (lats - lats[i])**2)
        closest_indices = np.argsort(distances)[1:n_closest+1]
        closest_coords = np.column_stack((lons[closest_indices], lats[closest_indices]))
        orientations[i] = np.mean(orientations[closest_indices], axis=0)
        vec = fit_linear_regression(closest_coords)
        all_vecs.append(vec)


    all_vecs = np.array(all_vecs)
    for i, vec in enumerate(all_vecs):
        _, angle, _ = geodesic.inv(lons[i], lats[i], lons[i]+0.001, lats[i]+0.001*vec[0])
        angle = (angle + 360) % 360
        avg_orientation = orientations[i]
        # avg_orientation = mean_angle(orientations[idx1:idx2])
        #cross_product = vec[0] * avg_orientation[1] - vec[1] * avg_orientation[0] 
        #angle_right = True if cross_product < 0 else False
        avg_orientation_angle = (geodesic.inv(lons[i], lats[i], lons[i]+0.001*avg_orientation[0], lats[i]+0.001*avg_orientation[1])[1] + 360) % 360
        #angle_right = abs((angle-90 - avg_orientation_angle + 180) % 360 - 180) < abs((angle+90 - avg_orientation_angle + 180) % 360 - 180)
        #angles.append((angle-90) % 360 if angle_right else (angle +90) % 360)
        angle = (find_closest_angle(angle, avg_orientation_angle) + 360) % 360

        angles.append(angle)
    plt.show()


        
    print(len(angles))
    return lons, lats, angles

















def remove_labels_from_size_thresholds(m, label1, label2, size_thr_1, size_thr_2):

    if size_thr_1:
        m_max = np.max(m)
        binary_map = np.isin(m, [label1])
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the thr
        for region_label in range(1, num_features + 1): # Skipping background (label 0)
            if region_sizes[region_label] < size_thr_1:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max
    if size_thr_2:
        m_max = np.max(m)
        binary_map = np.isin(m, [label2])
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the thr
        for region_label in range(1, num_features + 1): # Skipping background (label 0)
            if region_sizes[region_label] < size_thr_2:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max

    return m


def compute_boundary_coordinates_between_labels_2(m, lon_map, lat_map, label1, label2, max_distance_to_avg=None, size_threshold_1=None, size_threshold_2=None):
    lons = []
    lats = []
    angles = []


    m = remove_labels_from_size_thresholds(m, label1, label2, size_threshold_1, size_threshold_2)



    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == label1:
                neighbors = [
                      (i-1, j ), (i+1, j + 1), (i, j - 1), (i, j + 1)
                ]
                # neighbors = [
                #     (i - 1, j), (i + 1, j)
                # ]
                neighbors = [
                    (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                    (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
                ]

                for ni, nj in neighbors:
                    if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
                        if m[ni, nj] == label2:
                            # Interpolate lon/lat values for the boundary
                            interp_lon = (lon_map[i, j] + lon_map[ni, nj]) / 2
                            interp_lat = (lat_map[i, j] + lat_map[ni, nj]) / 2
                            lons.append(interp_lon)
                            lats.append(interp_lat)

                            # #Calculate the angle
                            angle,angle2,distance = geodesic.inv(lon_map[i, j], lat_map[i, j], lon_map[ni, nj], lat_map[ni, nj])
                            angle2 = (angle2 + 360) % 360

                            angles.append(angle2)






    if max_distance_to_avg != None:
        for i in range(len(lons)):
            sum = angles[i]
            tot_points = 1
            lons_filtered = [lons[i]]
            lats_filtered = [lats[i]]
            for j in range(len(lons)):
                if j != i:
                    _,_,distance = geodesic.inv(lons[i], lats[i], lons[j], lats[j])

                    if distance < max_distance_to_avg:
                        sum += angles[j]
                        tot_points += 1
                        lons_filtered.append(lons[j])
                        lats_filtered.append(lats[j])


            slope, intercept, _, _, _ = linregress(lons_filtered, lats_filtered)

            lon2 = lons_filtered[0] + 0.01
            lat2 = lats_filtered[0] + slope*0.01
            angle,angle2,distance = geodesic.inv(lons_filtered[0], lats_filtered[0], lon2, lat2)
            angle2 = (angle2 + 360) % 360 
            if sum / tot_points >= angle2:
                angle_normal = (angle2 + 90 +  360) % 360
            else: 
                angle_normal = (angle2 - 90 +  360) % 360

            # Calculate the angle 90 degrees normal to the regression line
            angle_normal = (sum / tot_points) 
            angles[i] =  (angle_normal + 360) % 360 #sum / tot_points
    return lons, lats, angles
    






# def compute_boundary_coordinates_between_labels(m, lon_map, lat_map, label1, label2):
#     lons = []
#     lats = []
#     highest_confidence = -1  # Initialize to a low value

#     for i in range(m.shape[0]):
#         for j in range(m.shape[1]):
#             if m[i, j] == label1:
#                 neighbors = [
#                     (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
#                 ]

#                 for ni, nj in neighbors:
#                     if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
#                         if m[ni, nj] == label2:
#                             # Calculate confidence (e.g., based on distance)
#                             confidence = (lon_map[i, j] - lon_map[ni, nj])**2 + (lat_map[i, j] - lat_map[ni, nj])**2

#                             if confidence > highest_confidence:
#                                 highest_confidence = confidence
#                                 # Store lon and lat values for the boundary
#                                 interp_lon = (lon_map[i, j] + lon_map[ni, nj]) / 2
#                                 interp_lat = (lat_map[i, j] + lat_map[ni, nj]) / 2
#                                 lons = [interp_lon]
#                                 lats = [interp_lat]
#     return lons, lats
