import numpy as np 
import xarray as xr 
import pandas as pd 

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
    
    # Initialize empty lists for data arrays
    # Create empty lists
    var_closed_list = []
    var_open_list = []
    var_border_list = []
    
    times_closed = []
    times_open = []
    times_border = []
    if var == "SSTOT":
        ss001_closed, ss001_open, ss001_border  = extract_var_at_idx(dict_list, "SS001")
        ss002_closed, ss002_open, ss002_border  = extract_var_at_idx(dict_list, "SS002")
        ss003_closed, ss003_open, ss003_border  = extract_var_at_idx(dict_list, "SS003")
        ss004_closed, ss004_open, ss004_border  = extract_var_at_idx(dict_list, "SS004")
        ss005_closed, ss005_open, ss005_border  = extract_var_at_idx(dict_list, "SS005")

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



        if "lev" in ds and lev_idx == None:
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
        elif "lev" in ds and lev_idx:       
            var_closed = np.concatenate(var_closed_list, axis=1)
            var_open = np.concatenate(var_open_list, axis=1)
            var_border = np.concatenate(var_border_list, axis=1)
            
            # You now need to make sure that 'lev' is the first axis (dimension) if it isn't already
            # We assume here that 'lev' exists and has the same index/length in all three datasets
            
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