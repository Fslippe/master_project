from pyhdf.SD import SD, SDC
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pyproj
import cartopy.crs as ccrs



# Define a function to convert Polar Stereographic coordinates to lon, lat
def transform_polar_stereo_to_lonlat(x, y):
    proj = pyproj.Transformer.from_crs(
        pyproj.CRS('EPSG:3411'),  # Polar Stereographic North
        pyproj.CRS('EPSG:4326'),  # WGS84 coordinate system
        always_xy=True
    )
    return proj.transform(x, y)


def get_water_mask(folder= "/uio/hume/student-u37/fslippe/data/land_sea_ice_mask/", filename="NSIDC-0780_SeaIceRegions_PS-N3.125km_v1.0.nc"):
    ds = (xr.open_dataset(folder + filename))

    # Get the x, y meshgrid from the dataset
    x, y = np.meshgrid(ds['x'], ds['y'])

    # Convert x, y to longitude and latitude
    longitude, latitude = transform_polar_stereo_to_lonlat(x, y)

    # Add the resulting longitude and latitude arrays to the dataset
    ds['longitude'] = (('y', 'x'), longitude)
    ds['latitude'] = (('y', 'x'), latitude)

    return ds
