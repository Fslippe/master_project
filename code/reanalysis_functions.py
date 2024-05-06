import numpy as np 
import xarray as xr 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor for regression tasks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import string 

def get_histogram_from_var_and_index(var, ix_tot, iy_tot, step_index, datetime_obj_list, date_list, ax, min_max_stds=None, v_max=None, y_bin_size=None, leg_loc="best"):
    m_vals = []
    for k in range(len(ix_tot)): 
        datetime_obj = datetime_obj_list[k]
        date = date_list[k]
        if len(ix_tot[k]) > 0:
            da = extract_var_at_indices(ix_tot[k].astype(int), iy_tot[k].astype(int), date, datetime_obj, var, lev_idx=67)
            m_vals.extend((da.values))
    tot_mean = np.nanmean(m_vals, axis=0)
    tot_median = np.nanmedian(m_vals, axis=0)


    if min_max_stds:
        m_vals = np.array(m_vals)
        m_vals_arr = m_vals
        var_std = np.nanstd(m_vals)
        var_mean = np.nanmean(m_vals)
        m_vals = np.where((m_vals >= var_mean - min_max_stds*var_std) & (m_vals <= var_mean + min_max_stds*var_std), m_vals, np.nan)
        m_vals_arr_masked = np.array(m_vals)
        ymin = np.nanmin(m_vals_arr_masked)
        ymax = np.nanmax(m_vals_arr_masked)
    else:
        m_vals_arr = np.array(m_vals)
        ymin = np.nanmin(m_vals_arr)
        ymax = np.nanmax(m_vals_arr)

    ax.plot(step_index, tot_mean, color="k", lw=3, alpha=0.5, label="mean")
    ax.plot(step_index, tot_median, color="b", lw=3, alpha=0.5, label="median")
    idx = np.argmin(step_index[:-1]-step_index[1:])
    step_index = np.insert(step_index, idx+1, 0)
    m_vals_arr = np.insert(m_vals_arr, idx+1, np.nan , axis=1)
    #step_index_xax = np.where(step_index < 0, step_index-16,step_index-16)


    mask = ~np.isnan(m_vals_arr)
    m_vals_masked = np.where(np.isnan(m_vals_arr), 0, m_vals_arr) # m_vals_arr[~np.isnan(m_vals_arr)]
    x = np.tile(step_index, (len(m_vals), 1))[mask].ravel()

    y = np.array(m_vals_masked)[mask].ravel()

    bins_x = len(step_index)

    try:
        y_bin_size.shape
        bins_y = y_bin_size
    except:   
        bins_y = 13
        binsize_y = (ymax-ymin) / bins_y  # Desired binsize for y-axis

        # Calculate the number of bins based on the binsize
        bins_y = int((np.max(y) - np.min(y)) / binsize_y)


    # Calculate the 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y])
    xedges = np.append(step_index-16, step_index[-1]+ 16)
    H_sum = H.sum(axis=1)  
    H = H / H_sum[:, np.newaxis] * 100 
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    H_masked = np.where(H <= 0, np.nan, H)
    if v_max == None:
        v_max = np.nanmax(H_masked)

    num_colors = v_max // 5 *2 #12

    cmap = colors.ListedColormap(plt.cm.GnBu(np.linspace(0, 1, num_colors)))
    cb = ax.imshow(H_masked.T, extent=extent, origin='lower', aspect='auto', cmap=cmap, vmax=v_max)
    plt.colorbar(cb, label="%")

    ax.set_xlim([step_index[0]-16, step_index[-1]+16])
    ax.set_ylim([ymin- (ymax-ymin)*0.05, ymax + (ymax-ymin)*0.15])
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    # ax.text(xmin + (xmax-xmin)*0.03, ymin + (ymax-ymin)*0.90, 'Closed Cell', bbox=dict(facecolor='tab:red', alpha=0.5), fontsize=14)
    # ax.text(xmin + (xmax-xmin)*0.35, ymin + (ymax-ymin)*0.90, 'Open Cell', bbox=dict(facecolor='tab:blue', alpha=0.5), fontsize=14)
    ax.text(xmin + (xmax-xmin)*0.275, ymin + (ymax-ymin)*0.90, 'Closed Cell', bbox=dict(facecolor='tab:red', alpha=0.5), fontsize=14)
    ax.text(xmin + (xmax-xmin)*0.515, ymin + (ymax-ymin)*0.90, 'Open Cell', bbox=dict(facecolor='tab:blue', alpha=0.5), fontsize=14)

    ax.set_title(string.capwords(da.attrs["long_name"].replace("_", " ")))
    ax.set_xlabel("step with wind [km]")
    ax.set_ylabel("[%s]" %da.attrs["units"])
    
    # Plotting data
    # Mark the line where step_index is 0
    ax.axvline(x=0, color='k', linestyle='--', label="border")
    ax.legend(loc=leg_loc)

    return H, xedges, yedges, H_sum




def get_potential_temperature(T, P, P0=1000):
    theta = T.values * (P0 / P[:, np.newaxis]) ** (0.286)
    theta_ds = xr.Dataset({"THETA": (('lev', 'time'), theta)},
                                coords={'time': T.time,
                                        'lev': T.lev.values})

    theta_ds.THETA.attrs["standard_name"] = "potential_air_temperature"
    theta_ds.THETA.attrs["long_name"] = "potential_air_temperature" 
    theta_ds.THETA.attrs["units"] = "K" 

    return theta_ds

def get_theta(T, P, P0=1000):
    theta = T * (P0 / P) ** (0.286)
    return theta 


def get_MCAOidx(Tskn, T850, Ps):
    theta850 = get_theta(T850, 850)
    thetaskn = get_theta(Tskn, Ps)
    return thetaskn - theta850  

def get_LTS(Ts, Ps, T700):  
    theta700 = get_theta(T700, 700)
    thetasurf = get_theta(Ts, Ps) 
    return theta700 - thetasurf 

def get_qs(T, p):
    e0 = 6.11 # hPa
    T0 = 273.15 # K 
    R = 8.314 # J mol−1 K−1
    lv = 40660 # J/mol

    es = e0 * np.exp(lv/R * (1/T0 - 1/T))
    return es / p

def get_moist_lapse_rate(T, p):
    g = 9.81 
    cp = 1006 # J/(kg*K) atm approx
    Ra = 287.052874 # J kg-1 K-1
    Rv = 461.5 # J kg-1 K-1
    lv = 2.257 # J/kg
    lapse_rate = g / cp * (1 -  (1 + lv*get_qs(T, p) / (Ra * T) / (1 + lv**2*get_qs(T, p) / (cp*Rv*T**2))))
    return lapse_rate

def get_pressure_height(T0, P0, P):
    Ra = 287.052874 # J kg-1 K-1
    g = 9.81 
    z = (Ra*T0 / g) * np.log(P0 / P)
    return z 

def get_EIS(T0, P0, T700, T850, LCL):
    LTS = get_LTS(T0, P0, T700)
    lapse_rate = get_moist_lapse_rate(T850, 850)
    z700 = get_pressure_height(T0, P0, 700)

    EIS = LTS - lapse_rate * (z700 - LCL)
    return EIS

def get_rf_dataframe(dict_list, vars, lev_h=67, lev_p=910, closed_lab=0, open_lab=2, border_lab=1):
    # Initialize empty lists to contain the data for each environment type
    data_closed = []
    data_open = []
    data_border = []
    
    # Extract and append the data for each variable
    for var in vars:
        closed, open_, border = extract_var_at_idx(dict_list, var)
        
        if "lev" in closed:
            lev_size = lev_p if var in ["U", "V", "QI", "QL", "T"] else lev_h      
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

    # Now, combine the three environment dataframes into one,
    labels_closed = [closed_lab] * len(df_closed)
    labels_open = [open_lab] * len(df_open)
    # each environment type will be a block of rows with the same index ranges
    if border_lab:
        df_border = pd.concat(data_border, axis=1)
        labels_border = [1] * len(df_border)
        df_combined = pd.concat([df_closed, df_open, df_border], axis=0)

    # Create the labels for each environment type
        labels = labels_closed + labels_open + labels_border
    else:
        df_combined = pd.concat([df_closed, df_open], axis=0)
        labels = labels_closed + labels_open

    # Create the labels DataFrame
    y = pd.DataFrame(labels, columns=['Label'])

    # Make sure we are only using the column names of the variables for the final DataFrame
    df_combined.columns = vars

    return df_combined.reset_index(drop=True), y
def align_time_coordinates(*data_arrays):
    # Align the data arrays along the time coordinate
    aligned = xr.align(*data_arrays, join='inner', copy=False)
    
    # Each element in 'aligned' tuple will be a data array with matching time coordinates
    return aligned

# Example usage:
def extract_var_at_idx(dict_list, var, lev_idx=None):
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
                    "CLDPRS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "PS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "T2M": "MERRA2_400.tavg1_2d_slv_Nx",
                    "TS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "T850": "MERRA2_400.tavg1_2d_slv_Nx",
                    "U10M": "MERRA2_400.tavg1_2d_slv_Nx",
                    "V10M": "MERRA2_400.tavg1_2d_slv_Nx",
                    "ZLCL": "MERRA2_400.tavg1_2d_slv_Nx",
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
    
    var_closed_list = []
    var_open_list = []
    var_border_list = []
    
    times_closed = []
    times_open = []
    times_border = []
    
    if var == "WIND10M":
        U10M_closed, U10M_open, U10M_border  = extract_var_at_idx(dict_list, "U10M")
        V10M_closed, V10M_open, V10M_border  = extract_var_at_idx(dict_list, "V10M")
        var_closed_ds = U10M_closed.copy()
        var_open_ds = U10M_open.copy()
        var_border_ds = U10M_border.copy()
        wind10m_closed = np.sqrt(U10M_closed.U10M.values**2 +V10M_closed.V10M.values**2)
        wind10m_open = np.sqrt(U10M_open.U10M.values**2 +V10M_open.V10M.values**2)
        wind10m_border = np.sqrt(U10M_border.U10M.values**2 +V10M_border.V10M.values**2)

        wind10m_closed_da = xr.DataArray(wind10m_closed, coords=U10M_closed.coords, dims=U10M_closed.dims)
        wind10m_open_da = xr.DataArray(wind10m_open, coords=U10M_open.coords, dims=U10M_open.dims)
        wind10m_border_da = xr.DataArray(wind10m_border, coords=U10M_border.coords, dims=U10M_border.dims)

        var_closed_ds["WIND10M"] = wind10m_closed_da.copy()
        var_open_ds["WIND10M"] = wind10m_open_da.copy()
        var_border_ds["WIND10M"] = wind10m_border_da.copy()

        var_closed_ds = var_closed_ds.drop_vars('U10M')
        var_open_ds = var_open_ds.drop_vars('U10M')
        var_border_ds = var_border_ds.drop_vars('U10M')

        var_closed_ds.WIND10M.attrs["standard_name"] = "10-meter_wind"
        var_closed_ds.WIND10M.attrs["long_name"] = "10-meter_wind"
        var_closed_ds.WIND10M.attrs["units"] = "m s-1"

        var_open_ds.WIND10M.attrs["standard_name"] = "10-meter_wind"
        var_open_ds.WIND10M.attrs["long_name"] = "10-meter_wind"
        var_open_ds.WIND10M.attrs["units"] = "m s-1"

        var_border_ds.WIND10M.attrs["standard_name"] = "10-meter_wind"
        var_border_ds.WIND10M.attrs["long_name"] = "10-meter_wind"
        var_border_ds.WIND10M.attrs["units"] = "m s-1"
        

    elif var == "SSTOT":
        ss001_closed, ss001_open, ss001_border  = extract_var_at_idx(dict_list, "SS001", lev_idx)
        ss002_closed, ss002_open, ss002_border  = extract_var_at_idx(dict_list, "SS002", lev_idx)
        ss003_closed, ss003_open, ss003_border  = extract_var_at_idx(dict_list, "SS003", lev_idx)
        ss004_closed, ss004_open, ss004_border  = extract_var_at_idx(dict_list, "SS004", lev_idx)
        ss005_closed, ss005_open, ss005_border  = extract_var_at_idx(dict_list, "SS005", lev_idx)

        # compute sttot
        var_closed_ds = ss001_closed.copy()
        var_open_ds = ss001_open.copy()
        var_border_ds = ss001_border.copy()
        sstot_closed = ss001_closed.SS001 + ss002_closed.SS002 + ss003_closed.SS003 + ss004_closed.SS004 + ss005_closed.SS005
        sstot_open = ss001_open.SS001 + ss002_open.SS002 + ss003_open.SS003 + ss004_open.SS004 + ss005_open.SS005
        sstot_border = ss001_border.SS001 + ss002_border.SS002 + ss003_border.SS003 + ss004_border.SS004 + ss005_border.SS005

        sstot_closed_da = xr.DataArray(sstot_closed, coords=ss001_closed.coords, dims=ss001_closed.dims)
        sstot_open_da = xr.DataArray(sstot_open, coords=ss001_open.coords, dims=ss001_open.dims)
        sstot_border_da = xr.DataArray(sstot_border, coords=ss001_border.coords, dims=ss001_border.dims)

        var_closed_ds["SSTOT"] = sstot_closed_da.copy()
        var_open_ds["SSTOT"] = sstot_open_da.copy()
        var_border_ds["SSTOT"] = sstot_border_da.copy()

        var_closed_ds = var_closed_ds.drop_vars('SS001')
        var_open_ds = var_open_ds.drop_vars('SS001')
        var_border_ds = var_border_ds.drop_vars('SS001')

        var_closed_ds.SSTOT.attrs["standard_name"] = "sea_salt_mixing_ratio"
        var_closed_ds.SSTOT.attrs["long_name"] = "sea_salt_mixing_ratio"
        var_closed_ds.SSTOT.attrs["units"] = "kg kg-1"

        var_open_ds.SSTOT.attrs["standard_name"] = "sea_salt_mixing_ratio"
        var_open_ds.SSTOT.attrs["long_name"] = "sea_salt_mixing_ratio"
        var_open_ds.SSTOT.attrs["units"] = "kg kg-1"

        var_border_ds.SSTOT.attrs["standard_name"] = "sea_salt_mixing_ratio"
        var_border_ds.SSTOT.attrs["long_name"] = "sea_salt_mixing_ratio"
        var_border_ds.SSTOT.attrs["units"] = "kg kg-1"

    elif var == "M":
        Tskn_c, Tskn_o, Tskn_b = extract_var_at_idx(dict_list, "TS")
        T850_c, T850_o, T850_b = extract_var_at_idx(dict_list, "T850")
        Ps_c, Ps_o, Ps_b = extract_var_at_idx(dict_list, "PS")
        var_closed_ds = Tskn_c.copy()
        var_open_ds = Tskn_o.copy()
        var_border_ds = Tskn_b.copy()

        m_data_closed = get_MCAOidx(Tskn_c.TS.values, T850_c.T850.values, Ps_c.PS.values/100)
        m_data_border = get_MCAOidx(Tskn_b.TS.values, T850_b.T850.values, Ps_b.PS.values/100)
        m_data_open = get_MCAOidx(Tskn_o.TS.values, T850_o.T850.values, Ps_o.PS.values/100)

        m_closed_da = xr.DataArray(m_data_closed, coords=Tskn_c.coords, dims=Tskn_c.dims)
        m_open_da = xr.DataArray(m_data_open, coords=Tskn_o.coords, dims=Tskn_o.dims)
        m_border_da = xr.DataArray(m_data_border, coords=Tskn_b.coords, dims=Tskn_b.dims)

        var_closed_ds["M"] = m_closed_da
        var_open_ds["M"] = m_open_da
        var_border_ds["M"] = m_border_da

        var_closed_ds = var_closed_ds.drop_vars('TS')
        var_open_ds = var_open_ds.drop_vars('TS')
        var_border_ds = var_border_ds.drop_vars('TS')

        var_closed_ds.M.attrs["standard_name"] = "MCAO_index"
        var_closed_ds.M.attrs["long_name"] = "marine_cold_air_outbreak_index"
        var_closed_ds.M.attrs["units"] = "K"

        var_open_ds.M.attrs["standard_name"] = "MCAO_index"
        var_open_ds.M.attrs["long_name"] = "marine_cold_air_outbreak_index"
        var_open_ds.M.attrs["units"] = "K"

        var_border_ds.M.attrs["standard_name"] = "MCAO_index"
        var_border_ds.M.attrs["long_name"] = "marine_cold_air_outbreak_index"
        var_border_ds.M.attrs["units"] = "K"

    elif var == "EIS":
        T0_c, T0_o, T0_b = extract_var_at_idx(dict_list, "T2M")
        T850_c, T850_o, T850_b = extract_var_at_idx(dict_list, "T850")
        Ps_c, Ps_o, Ps_b = extract_var_at_idx(dict_list, "PS")
        T700_c, T700_o, T700_b = extract_var_at_idx(dict_list, "T", lev_idx=700)
        LCL_c, LCL_o, LCL_b = extract_var_at_idx(dict_list, "ZLCL", lev_idx=700)
        T0_c, T850_c, Ps_c, T700_c, LCL_c = align_time_coordinates(T0_c, T850_c, Ps_c, T700_c, LCL_c)
        T0_o, T850_o, Ps_o, T700_o, LCL_o = align_time_coordinates(T0_o, T850_o, Ps_o, T700_o, LCL_o)
        T0_b, T850_b, Ps_b, T700_b, LCL_b = align_time_coordinates(T0_b, T850_b, Ps_b, T700_b, LCL_b)

        var_closed_ds = T0_c.copy()
        var_open_ds = T0_o.copy()
        var_border_ds = T0_b.copy()

        eis_data_closed = get_EIS(T0_c.T2M.values, Ps_c.PS.values/100, T700_c.T.values, T850_c.T850.values, LCL_c.ZLCL.values)
        eis_data_border = get_EIS(T0_b.T2M.values, Ps_b.PS.values/100, T700_b.T.values, T850_b.T850.values, LCL_b.ZLCL.values)
        eis_data_open = get_EIS(T0_o.T2M.values, Ps_o.PS.values/100, T700_o.T.values, T850_o.T850.values, LCL_o.ZLCL.values)

        eis_closed_da = xr.DataArray(eis_data_closed, coords=T0_c.coords, dims=T0_c.dims)
        eis_open_da = xr.DataArray(eis_data_open, coords=T0_o.coords, dims=T0_o.dims)
        eis_border_da = xr.DataArray(eis_data_border, coords=T0_b.coords, dims=T0_b.dims)

        var_closed_ds['EIS'] = eis_closed_da
        var_open_ds['EIS'] = eis_open_da
        var_border_ds['EIS'] = eis_border_da

        var_closed_ds = var_closed_ds.drop_vars('T2M')
        var_open_ds = var_open_ds.drop_vars('T2M')
        var_border_ds = var_border_ds.drop_vars('T2M')

        var_closed_ds.EIS.attrs["standard_name"] = "EIS_index"
        var_closed_ds.EIS.attrs["long_name"] = "estimated_inversion_strength"
        var_closed_ds.EIS.attrs["units"] = "K"

        var_open_ds.EIS.attrs["standard_name"] = "EIS_index"
        var_open_ds.EIS.attrs["long_name"] = "estimated_inversion_strength"
        var_open_ds.EIS.attrs["units"] = "K"

        var_border_ds.EIS.attrs["standard_name"] = "EIS_index"
        var_border_ds.EIS.attrs["long_name"] = "estimated_inversion_strength"
        var_border_ds.EIS.attrs["units"] = "K"
    else:
        for dic in dict_list:
            date = dic["date"]
            try:
                ds = xr.open_dataset(f"/uio/hume/student-u37/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc")
            except:
                print(f"/uio/hume/student-u37/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc")
                continue 
            try:
                time_sel = ds[var].sel(time=dic["datetime"], method="nearest")
            except:
                print(f"/uio/hume/student-u37/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc")

                continue

            for condition, var_list, times_list in zip(
                    ["idx_closed", "idx_open", "idx_border"], 
                    [var_closed_list, var_open_list, var_border_list],
                    [times_closed, times_open, times_border]):

                if condition in dic and len(dic[condition]) > 0: 
                    if "lev" in ds:
                        var_list.append(time_sel.values[:, dic[condition][:, 0], dic[condition][:, 1]])

                    else:
                        var_list.append(time_sel.values[dic[condition][:, 0], dic[condition][:, 1]])
                    times_list.extend([dic["datetime"]] * len(dic[condition]))



        if "lev" in ds and lev_idx == None:
            # Use axis=1 if you're concatenating 3D data along the 'lev' dimension
            var_closed = np.concatenate(var_closed_list, axis=1)
            var_open = np.concatenate(var_open_list, axis=1)
            var_border = np.concatenate(var_border_list, axis=1)
            
            
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

        elif "lev" in ds and lev_idx:       
            var_closed = np.concatenate(var_closed_list, axis=1)
            var_open = np.concatenate(var_open_list, axis=1)
            var_border = np.concatenate(var_border_list, axis=1)
            
            # Create the dataset with the 'lev' coordinate
            var_closed_ds = xr.Dataset({var: (('lev', 'time'), var_closed)},
                                    coords={'time': times_closed,
                                            'lev': ds.lev.values}).sel(lev=lev_idx)  # Make sure to extract the level values from ds
            var_open_ds = xr.Dataset({var: (('lev', 'time'), var_open)},
                                    coords={'time': times_open,
                                            'lev': ds.lev.values}).sel(lev=lev_idx)
            var_border_ds = xr.Dataset({var: (('lev', 'time'), var_border)},
                                    coords={'time': times_border,
                                            'lev': ds.lev.values}).sel(lev=lev_idx)

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


def extract_var_at_indices(ix, iy, date, datetime_t, var, lev_idx=None):
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
                    "CLDPRS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "PS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "T2M": "MERRA2_400.tavg1_2d_slv_Nx",
                    "TS": "MERRA2_400.tavg1_2d_slv_Nx",
                    "T850": "MERRA2_400.tavg1_2d_slv_Nx",
                    "U10M": "MERRA2_400.tavg1_2d_slv_Nx",
                    "V10M": "MERRA2_400.tavg1_2d_slv_Nx",
                    "ZLCL": "MERRA2_400.tavg1_2d_slv_Nx",
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
    
    var_list = []
    
    times = []
    
    if var == "WIND10M":
        U10M = extract_var_at_indices(ix, iy, date, datetime_t, "U10M")
        V10M = extract_var_at_indices(ix, iy, date, datetime_t, "V10M")
        var_arr = np.sqrt(U10M.values**2 + V10M.values**2)
        var_da = xr.DataArray(var_arr) 

        var_da.attrs["standard_name"] = "10-meter_wind"
        var_da.attrs["long_name"] = "10-meter_wind"
        var_da.attrs["units"] = "m s-1"
        
    elif var == "SSTOT":
        ss001 = extract_var_at_indices(ix, iy, date, datetime_t, "SS001", lev_idx)
        ss002 = extract_var_at_indices(ix, iy, date, datetime_t, "SS002", lev_idx)
        ss003 = extract_var_at_indices(ix, iy, date, datetime_t, "SS003", lev_idx)
        ss004 = extract_var_at_indices(ix, iy, date, datetime_t, "SS004", lev_idx)
        ss005 = extract_var_at_indices(ix, iy, date, datetime_t, "SS005", lev_idx)

        # compute sttot
        var_arr = ss001 + ss002 + ss003 + ss004 + ss005
        var_da = xr.DataArray(var_arr) 

        var_da.attrs["standard_name"] = "sea_salt_mixing_ratio"
        var_da.attrs["long_name"] = "sea_salt_mixing_ratio"
        var_da.attrs["units"] = "kg kg-1"

     

    elif var == "M":
        Tskn = extract_var_at_indices(ix, iy, date, datetime_t, "TS")
        T850 = extract_var_at_indices(ix, iy, date, datetime_t, "T850")
        Ps = extract_var_at_indices(ix, iy, date, datetime_t, "PS")

        var_arr = get_MCAOidx(Tskn, T850, Ps/100)
        var_da = xr.DataArray(var_arr) 

        var_da.attrs["standard_name"] = "MCAO_index"
        var_da.attrs["long_name"] = "marine_cold_air_outbreak_index"
        var_da.attrs["units"] = "K"


    elif var == "EIS":
        T0 = extract_var_at_indices(ix, iy, date, datetime_t, "T2M")
        T850 = extract_var_at_indices(ix, iy, date, datetime_t, "T850")
        Ps = extract_var_at_indices(ix, iy, date, datetime_t, "PS")
        T700 = extract_var_at_indices(ix, iy, date, datetime_t, "T", lev_idx=700)
        LCL = extract_var_at_indices(ix, iy, date, datetime_t, "ZLCL")


        var_arr = get_EIS(T0, Ps/100, T700, T850, LCL)
        var_da = xr.DataArray(var_arr) 

        var_da.attrs["standard_name"] = "EIS_index"
        var_da.attrs["long_name"] = "estimated_inversion_strength"
        var_da.attrs["units"] = "K"

    else:
        if lev_idx:
            try:
                ds = xr.open_dataset(f"/uio/hume/student-u37/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc").sel(lev=lev_idx)
            except:
                ds = xr.open_dataset(f"/uio/hume/student-u37/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc")

            var_arr = ds[var].sel(time=datetime_t, method="nearest").values

        else:
            ds = xr.open_dataset(f"/uio/hume/student-u37/fslippe/MERRA/{date[:4]}/{lookup_table[var]}.{date}.SUB.nc")
            var_arr = ds[var].sel(time=datetime_t, method="nearest").values

      # Creates a boolean array indicating True where ix is valid and False where it is -9999
        mask_ix = ix != -9999
        # Initialize an array of the same length as ix filled with np.nan
        result = np.full(ix.shape, np.nan)

        if var_arr.ndim == 3 and lev_idx == None:
            print(ds)
            # Only apply the valid indices to the result where ix has valid values
            print(var_arr.shape)
            result[mask_ix] = var_arr[:, ix[mask_ix], iy[mask_ix]]

            var_list.append(result)
            var_arr = np.concatenate(var_list, axis=1)

        elif var_arr.shape[0] != 45:
            return var_arr

        else:
            # Same process for the case without "lev"
            result[mask_ix] = var_arr[ix[mask_ix], iy[mask_ix]]

            var_list.append(result)

            var_arr = np.concatenate(var_list, axis=1)
        
        var_da = xr.DataArray(var_arr) 
        var_da.attrs = ds[var].attrs

    return var_da