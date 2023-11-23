import os
import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import matplotlib as mpl
import sys
import matplotlib.path as mplPath

### python /uio/hume/student-u37/fslippe/master_project/code/label_transitions.py
    

def on_press(event):
    global drawing, released
    zooming_panning = fig.canvas.toolbar.mode
    if zooming_panning == "":
        drawing = True
        released = False  # Update the global variable
    else:
        drawing = False


def on_motion(event):
    global last_point, released
    # Append the position (x, y) to the coords list only if drawing is True
    if event.inaxes == ax and drawing:
        current_point = (event.xdata, event.ydata)
        if last_point:
            coords.append((current_point[0], current_point[1]))
            connect.append(True)
        elif last_point is None and current_point:
            coords.append((current_point[0], current_point[1]))
            connect.append(False)

        # Update the last point
        last_point = current_point

def on_release(event):
    global drawing, last_point, released
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


def interpolate_coords(coords, connect):
    """Interpolate between points in coords if they are not neighbors."""
    interpolated = []
    for i in range(len(coords) - 1):
        if connect[i+1]:
            start = coords[i]
            end = coords[i + 1]
            # Check if points are neighbors
            if max(abs(start[0] - end[0]), abs(start[1] - end[1])) > 1:
                interpolated.extend(bresenham_line(
                    round(start[0]), round(start[1]), round(end[0]), round(end[1])))
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


def draw_area(fig, ax, data, mask_type, coords, folder_save_masks, folder_save_coords):
    print(f"Please draw the {mask_type}")
    ax.imshow(data, cmap='gray')

    # Connect the functions to the relevant events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    plt.show()

    mask = np.zeros(data.shape)  # Initialize the mask with zeros
    if len(coords) != 0:
        coords = interpolate_coords(coords, connect)
        coord_array = np.array([(round(coord[0]), round(coord[1])) for coord in coords])
        brush = gaussian_brush(width=50, height=50, sigma=15)

        for coord in coords:
            mask = apply_brush(mask, round(coord[0]), round(coord[1]), brush)

    plt.ion()
    plt.figure(figsize=(10, 10))  # Create a new figure explicitly
    plt.imshow(data, cmap='gray')
    plt.imshow(mask, alpha=0.3, cmap='Reds')
    run = input(f"Are you happy with the {mask_type} result? (y/n/q): ")
    plt.show()
    plt.ioff()

    if run == "y":
        plt.close()
        np.save(os.path.join(folder_save_coords, f"{file}_coords_{mask_type.lower()}"), coord_array)
        np.save(os.path.join(folder_save_masks, f"{file}_mask_{mask_type.lower()}"), mask)
    elif run == "n":
        plt.close()
        coords.clear()
    elif run == "q":
        sys.exit()

    return run

    




# Assuming your image is in variable `img`
# Initialize last_point as None
print("\n### Thank you for helping! ###")
print("Please draw a transition and close the figure when finished. You can use the zoom functionality to get a closer look at the picture")

name = input("Please enter your name: ")
if name == "":
    name = input("Please enter your name: ")


folder = "/scratch/fslippe/label_data/modis_data/"
all_files = os.listdir(folder)
npy_files = [f for f in all_files if f.endswith('.npy')]# and f[:-4] + ".npy" not in os.listdir("/scratch/fslippe/modis/MOD02/cao_test_data/")]

folder_save_masks = os.path.join("/".join(folder.split("/")[:-2]), "masks", name)
folder_save_coords = os.path.join("/".join(folder.split("/")[:-2]), "coords", name)

print(folder_save_coords)
print(folder_save_masks)

if not os.path.exists(folder_save_masks):
    os.makedirs(folder_save_masks)
if not os.path.exists(folder_save_coords):
    os.makedirs(folder_save_coords)




import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def fill_area_manually(coords, data_shape):
    coords = [(round(c[1]), round(c[0])) for c in coords]

    # Create a path object using the coordinates
    path = mplPath.Path(coords)

    # create an empty binary matrix with same shape as the image data
    mask = np.zeros(data_shape)

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if path.contains_point((r,c)):
                mask[r,c] = 1  # Set to 1 if the point is within the path
    return mask


last_point = None

def run_again(file, coords, fig, ax):
    global folder_save_masks, folder_save_coords
    
    filepath = os.path.join(folder, file)  # Full path to the file
    data = np.load(filepath)
    
    ax.imshow(data, cmap='gray')

    # Connect the functions to the relevant events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    plt.show()

    mask_cao = np.zeros(data.shape)  # Initialize the mask with zeros
    cord_img = np.zeros(data.shape)  # Initialize the mask with zeros

    brush = gaussian_brush(width=50, height=50, sigma=15)
    if len(coords) != 0:
        coords = interpolate_coords(coords, connect)
        coord_array = np.array([(round(coord[0]), round(coord[1])) for coord in coords])
        # for idx in coord_array:
        #     cord_img[idx[1], idx[0]] = 1
        
        for coord in coords:
            mask_cao = apply_brush(mask_cao, round(coord[0]), round(
                coord[1]), brush)  # Note the reversed indices
        # Run the create_path function here
       # mask_patch_cao = fill_area_manually(coords, data.shape)
        #create_path(folder_save_coords)

        

    plt.ion()
    plt.figure(figsize=(10, 10))  # Create a new figure explicitly
    plt.imshow(data, cmap='gray')
    
    plt.imshow(mask_cao, alpha=0.3, cmap='Reds')
    run = input(f"Are you happy with the result? (y/n/q): ")
    plt.show()
    plt.ioff()

    if run == "y":
        plt.close()

        # Save mask and coords
        np.save(os.path.join(folder_save_coords, f"{file}_coords_{mask_type}"), coord_array)
        np.save(os.path.join(folder_save_masks, f"{file}_mask_{mask_type}"), mask)

        # Check if the user wants to fill the area inside
        mask_filled_type = f"{mask_type}_filled"
        mask_filled = np.zeros(data.shape)

        plt.ion()
        plt.figure(figsize=(10, 10))
        #plt.imshow(data, cmap='gray')
        plt.imshow(mask_filled, alpha=0.3, cmap='Reds')
        fill = input(f"Do you want to fill the area inside the {mask_type_capitalized}? (y/n): ")
        plt.show()
        plt.ioff()

        if fill == "y":
            plt.close()
            mask_filled[mask_filled > 0] = 1  # Set all non-zero values to 1
            np.save(os.path.join(folder_save_masks, f"{file}_mask_{mask_filled_type}"), mask_filled)

    elif run == "n":
        plt.close()
        coords.clear()
    elif run == "q":
        sys.exit()






    print("Please draw a transition line")

    # ax.imshow(data, cmap='gray')

    # # Connect the functions to the relevant events
    # fig.canvas.mpl_connect('button_press_event', on_press)
    # fig.canvas.mpl_connect('button_release_event', on_release)
    # fig.canvas.mpl_connect('motion_notify_event', on_motion)
    # plt.show()
    # mask_transition = np.zeros(data.shape)  # Initialize the mask_transition with zeros
    # if len(coords) != 0:
    #     coords = interpolate_coords(coords, connect)
    #     coord_array_transition = np.array([(round(coord[0]), round(coord[1])) for coord in coords])
    #     brush = gaussian_brush(width=50, height=50, sigma=15)
        
    #     for coord in coords:
    #         if round(coord[0]) != int(coord[0]):
    #             print(round(coord[0]), int(coord[0])) 
    #         mask_transition = apply_brush(mask_transition, round(coord[0]), round(
    #             coord[1]), brush)  # Note the reversed indices


    # plt.ion() 
    # print(coord_array_transition)
    # plt.figure(figsize=(10, 10))  # Create a new figure explicitly
    # plt.imshow(data, cmap='gray')
    # plt.imshow(mask_transition, alpha=0.3, cmap='Blues')
    # run = input("Are you happy with the result y/n or quit q: ")
    # plt.show()
    # plt.ioff()  # Turn on interactive mode again for subsequent plots

    # if run == "y":
    #     #np.save("%s/data/%s" % (folder_save, file[:-4]), data)
    #     plt.close()
    #     np.save(folder_save_masks + "/" + file + "_mask_transition", mask_transition)
    #     np.save(folder_save_coords + "/" + file + "_coords_transition", coord_array_transition)


    # elif run == "n":
    #     plt.close()
    #     coords.clear()  
    #     #run_again(file, coords, fig, ax)
    # elif run == "q":
    #     sys.exit()
    
    return run


for file in npy_files:
    while True:
        fig, ax = plt.subplots(figsize=(10, 10))
        coords = []
        connect = []
        # Flag to check if the mouse button is pressed
        drawing = False
        released = False
        run = run_again(file, coords, fig, ax)
        if run == "y":
            break
        elif run == "q":
            sys.exit()
