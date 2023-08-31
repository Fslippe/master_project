import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import convolve2d

from pyhdf.SD import SD, SDC
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap, cm


def on_press(event):
    global drawing
    zooming_panning = fig.canvas.toolbar.mode
    if zooming_panning == "":
        drawing = True
    else:
        drawing = False


def on_motion(event):
    global last_point
    # Append the position (x, y) to the coords list only if drawing is True
    if event.inaxes == ax and drawing:
        current_point = (event.xdata, event.ydata)

        # If last_point exists, interpolate between last_point and current_point
        if last_point:
            # Use linear interpolation
            x_coords = np.linspace(last_point[0], current_point[0], int(
                np.abs(current_point[0] - last_point[0])))
            y_coords = np.linspace(last_point[1], current_point[1], int(
                np.abs(current_point[1] - last_point[1])))

            for x, y in zip(x_coords, y_coords):
                coords.append((x, y))
                ax.scatter(x, y, color='red', s=5)

        # Update the last point
        last_point = current_point


def on_release(event):
    global drawing, last_point
    drawing = False
    last_point = None  # Reset last_point on release


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


def interpolate_coords(coords):
    """Interpolate between points in coords if they are not neighbors."""
    interpolated = []
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        # Check if points are neighbors
        if max(abs(start[0] - end[0]), abs(start[1] - end[1])) > 1:
            interpolated.extend(bresenham_line(
                int(start[0]), int(start[1]), int(end[0]), int(end[1])))
        else:
            interpolated.append(start)
    interpolated.append(coords[-1])  # Add the last point
    return interpolated

# ... [your existing code to collect coords]


def gaussian_brush(width=5, height=5, sigma=1.0):
    """
    Create a 2D Gaussian brush centered in the middle of the width and height.
    """
    x, y = np.meshgrid(np.linspace(-width//2, width//2, width),
                       np.linspace(-height//2, height//2, height))
    d = np.sqrt(x*x + y*y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
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
    return mask


# Assuming your image is in variable `img`
# Initialize last_point as None
folder = "/home/filip/Downloads/outbreak/"
all_files = os.listdir(folder)
hdf_files = [f for f in all_files if f.endswith(
    '.hdf') and f[:-4] + ".npy" not in os.listdir("/home/filip/Documents/master_project/training_set/data/")]


last_point = None

for file in hdf_files:
    folder_save = "training_set"
    fig, ax = plt.subplots(figsize=(10, 10))
    filepath = os.path.join(folder, file)  # Full path to the file
    hdf = SD(filepath, SDC.READ)

    data = hdf.select("1km Surface Reflectance Band 1")[:]
    # List to store the mouse positions
    coords = []

    # Flag to check if mouse button is pressed
    drawing = False

    ax.imshow(data, cmap='gray')

    # Connect the functions to the relevant events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()
    mask = np.zeros(data.shape)  # Initialize the mask with zeros
    if len(coords) != 0:
        coords = interpolate_coords(coords)

        brush = gaussian_brush(width=50, height=50, sigma=15)

        for coord in coords:
            mask = apply_brush(mask, int(coord[0]), int(
                coord[1]), brush)  # Note the reversed indices

    # avg_grid_size = 9
    # kernel = np.ones((avg_grid_size, avg_grid_size)) * 1/avg_grid_size**2
    # smoothed_output = convolve2d(mask, kernel),
    # print(smoothed_output[0].shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data, cmap='gray')
    ax.imshow(mask, alpha=0.3, cmap='Reds')
    print(np.max(mask))
    plt.show()
    # plt.imshow(mask, cmap="gray")
    # Print coords after closing the plot
    # plt.show()

    # arr = np.array(coords)
    np.save("%s/data/%s" % (folder_save, file[:-4]), data)

    np.save("%s/mask/%s_coords" % (folder_save, file[:-4]), mask)
