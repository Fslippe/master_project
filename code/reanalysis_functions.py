import numpy as np 
import xarray as xr 
import pandas as pd 

def get_potential_temperature(T, P, P0):
    theta = T.values * (P0 / P[:, np.newaxis]) ** (0.286)
    theta_ds = xr.Dataset({"THETA": (('lev', 'time'), theta)},
                                coords={'time': T.time,
                                        'lev': T.lev.values})

    theta_ds.THETA.attrs["standard_name"] = "potential_air_temperature"
    theta_ds.THETA.attrs["long_name"] = "potential_air_temperature" 
    theta_ds.THETA.attrs["units"] = "K" 

    return theta_ds

def get_LTS():  

def get_qs(T, p):

def get_moist_lapse_rate():
    g = 9.81 
    cp = 
    lapse_rate = g / cp * (1 -  (1 + lv*get_qs(T, p) / (Ra * T) / (1 + lv**2*get_qs(T, p) / (cp*Rv*T**2))))

def get_EIS():

def get_rf_dataframe(dict_list, vars, lev_h=67, lev_p=910):
    # Initialize empty lists to contain the data for each environment type
    data_closed = []
    data_open = []
    data_border = []
    
    # Extract and append the data for each variable
    for var in vars:
        if var == "SStot":
            ss001_closed, ss001_open, ss001_border  = extract_var_at_idx(dict_list, "SS001")
            ss002_closed, ss002_open, ss002_border  = extract_var_at_idx(dict_list, "SS002")
            ss003_closed, ss003_open, ss003_border  = extract_var_at_idx(dict_list, "SS003")
            ss004_closed, ss004_open, ss004_border  = extract_var_at_idx(dict_list, "SS004")
            ss005_closed, ss005_open, ss005_border  = extract_var_at_idx(dict_list, "SS005")
            sstot_closed = ss001_closed.SS001 + ss002_closed.SS002 + ss003_closed.SS003 + ss004_closed.SS004 + ss005_closed.SS005
            sstot_open = ss001_open.SS001 + ss002_open.SS002 + ss003_open.SS003 + ss004_open.SS004 + ss005_open.SS005
            sstot_border = ss001_border.SS001 + ss002_border.SS002 + ss003_border.SS003 + ss004_border.SS004 + ss005_border.SS005
            lev_size = lev_h       
            print(sstot_closed.lev)
            data_closed.append(sstot_closed.sel(lev=lev_size).to_series())
            data_open.append(sstot_open.sel(lev=lev_size).to_series())
            data_border.append(sstot_border.sel(lev=lev_size).to_series())
            
        else:
            closed, open_, border = extract_var_at_idx(dict_list, var)
            
            if "lev" in closed:
                lev_size = lev_p if var in ["U", "V", "QI", "QL"] else lev_h      
                data_closed.append(closed[var].sel(lev=lev_size).to_series())
                data_open.append(open_[var].sel(lev=lev_size).to_series())
                data_border.append(border[var].sel(lev=lev_size).to_series())
            else:
                data_closed.append(closed[var].to_series())
                data_open.append(open_[var].to_series())
                data_border.append(border[var].to_series())

    # Concatenate the data for each environment type
    df_closed = pd.concat(data_closed, axis=1)
    df_open = pd.concat(data_open, axis=1)
    df_border = pd.concat(data_border, axis=1)

    # Now, combine the three environment dataframes into one,
    # each environment type will be a block of rows with the same index ranges
    df_combined = pd.concat([df_closed, df_open, df_border], axis=0)

    # Create the labels for each environment type
    labels_closed = [0] * len(df_closed)
    labels_open = [0] * len(df_open)
    labels_border = [1] * len(df_border)
    labels = labels_closed + labels_open + labels_border

    # Create the labels DataFrame
    y = pd.DataFrame(labels, columns=['Label'])

    # Make sure we are only using the column names of the variables for the final DataFrame
    df_combined.columns = vars

    return df_combined.reset_index(drop=True), y

def extract_var_at_idx(dict_list, var):
    lookup_table = {"U": "MERRA2.wind_at_950hpa", 
                    "V": "MERRA2.wind_at_950hpa",
                    "AIRDENS": "MERRA2_400.inst3_3d_aer_Nv",
                    "SO4": "MERRA2_400.inst3_3d_aer_Nv",
                    "SS001": "MERRA2_400.inst3_3d_aer_Nv",
                    "SS002": "MERRA2_400.inst3_3d_aer_Nv",
                    "SS003": "MERRA2_400.inst3_3d_aer_Nv",
                    "SS004": "MERRA2_400.inst3_3d_aer_Nv",
                    "SS005": "MERRA2_400.inst3_3d_aer_Nv",
                    "CLDTMP": "MERRA2_400.tavg1_2d_slv_Nx",
                    "TQL": "MERRA2_400.tavg1_2d_slv_Nx",
                    "TQV": "MERRA2_400.tavg1_2d_slv_Nx",
                    "TQI": "MERRA2_400.tavg1_2d_slv_Nx",
                    "TS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "QL":"MERRA2_400.inst3_3d_asm_Np",
                    "QI":"MERRA2_400.inst3_3d_asm_Np",
                    "T":"MERRA2_400.inst3_3d_asm_Np",
                    "H":"MERRA2_400.inst3_3d_asm_Np",
                    "PBLH": "MERRA2_400.tavg1_2d_flx_Nx",
                    "PRECTOT": "MERRA2_400.tavg1_2d_flx_Nx",
                    "PRECTOTCORR": "MERRA2_400.tavg1_2d_flx_Nx"
                    }
    
    # Initialize empty lists for data arrays
# Create empty lists
    var_closed_list = []
    var_open_list = []
    var_border_list = []
    
    times_closed = []
    times_open = []
    times_border = []

    for dic in dict_list:
        date = dic["date"]
        ds = xr.open_dataset(f"/scratch/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc")
        time_sel = ds[var].sel(time=dic["datetime"], method="nearest")

        for condition, var_list, times_list in zip(
                ["idx_closed", "idx_open", "idx_border"], 
                [var_closed_list, var_open_list, var_border_list],
                [times_closed, times_open, times_border]):

            if condition in dic and len(dic[condition]) > 0:  # Ensure the idx array is not empty
                if "lev" in ds:
                    var_list.append(time_sel.values[:, dic[condition][:, 0], dic[condition][:, 1]])

                else:
                    var_list.append(time_sel.values[dic[condition][:, 0], dic[condition][:, 1]])
                times_list.extend([dic["datetime"]] * len(dic[condition]))



    if "lev" in ds:
        # Use axis=1 if you're concatenating 3D data along the 'lev' dimension
        var_closed = np.concatenate(var_closed_list, axis=1)
        var_open = np.concatenate(var_open_list, axis=1)
        var_border = np.concatenate(var_border_list, axis=1)
        
        # You now need to make sure that 'lev' is the first axis (dimension) if it isn't already
        # We assume here that 'lev' exists and has the same index/length in all three datasets
        
        # Create the dataset with the 'lev' coordinate
        var_closed_ds = xr.Dataset({var: (('lev', 'time'), var_closed)},
                                coords={'time': times_closed,
                                        'lev': ds.lev.values})  # Make sure to extract the level values from ds
        var_open_ds = xr.Dataset({var: (('lev', 'time'), var_open)},
                                coords={'time': times_open,
                                        'lev': ds.lev.values})
        var_border_ds = xr.Dataset({var: (('lev', 'time'), var_border)},
                                coords={'time': times_border,
                                        'lev': ds.lev.values})
                                
    else:
        var_closed = np.concatenate(var_closed_list, axis=0)
        var_open = np.concatenate(var_open_list, axis=0)
        var_border = np.concatenate(var_border_list, axis=0)
        var_closed_ds = xr.Dataset({var: (('time',), var_closed)},
                                coords={'time': times_closed})
        var_open_ds = xr.Dataset({var: (('time',), var_open)},
                                coords={'time': times_open})
        var_border_ds = xr.Dataset({var: (('time',), var_border)},
                                coords={'time': times_border})

    var_closed_ds.attrs.update(ds.attrs)
    var_open_ds.attrs.update(ds.attrs)
    var_border_ds.attrs.update(ds.attrs)
    var_closed_ds[var].attrs.update(ds[var].attrs)
    var_open_ds[var].attrs.update(ds[var].attrs)
    var_border_ds[var].attrs.update(ds[var].attrs)
    return var_closed_ds, var_open_ds, var_border_ds