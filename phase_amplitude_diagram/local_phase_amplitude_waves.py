import numpy as np
import xarray as xr
import multiprocessing as mp

def compute_amplitude_and_phase(A: xr.DataArray, stdA, std_dA) -> xr.Dataset:
    """
    Compute amplitude, phase, and angle of scatter points from a time-series data array.

    Parameters:
    - A (xr.DataArray): Input time-series data array.
    - stdA: Standard deviation of A over all times, lons and lats.
    - std_dA: Standard deviation of dA/dt over all times, lons and lats.

    Returns:
    - xr.Dataset: Dataset containing amplitude, phase, and angle of scatter points.
    """

    # Normalize A and its derivative along time
    A_norm = (A - A.mean()) / stdA
    B_norm = A.differentiate('time')
    B_norm = (B_norm - B_norm.mean()) / std_dA
    
    # Compute distance of every scatter point from the center
    amplitude = np.sqrt(A_norm**2 + B_norm**2)
    
    # Compute angle theta of every scatter point with respect to the positive x-axis in degrees
    theta = np.arctan2(B_norm, A_norm) * 180 / np.pi
    
    # Shift the angle range from [-180, 180] to [0, 360]
    theta[theta < 0] += 360
    
    # Classify the points to belong to a phase from 1 to 8
    phase = np.zeros_like(theta, dtype=int)
    phase[(theta >= 22.5) & (theta < 67.5)] = 4
    phase[(theta >= 67.5) & (theta < 112.5)] = 3
    phase[(theta >= 112.5) & (theta < 157.5)] = 2
    phase[(theta >= 157.5) & (theta < 202.5)] = 1
    phase[(theta >= 202.5) & (theta < 247.5)] = 8
    phase[(theta >= 247.5) & (theta < 292.5)] = 7
    phase[(theta >= 292.5) & (theta < 337.5)] = 6
    phase[(theta >= 337.5) | (theta < 22.5)] = 5
    
    # Create an xarray Dataset with 3 variables, amplitude, phase, and theta, all of dimension time
    ds = xr.Dataset(
        {
            "amplitude": amplitude,
            "theta": theta,
            "phase": xr.DataArray(phase, dims="time", coords={"time": A.time}),
        },
        coords={"time": A.time},
    )
    
    return ds


def process_lat_lon(data, lat_lon, stdA, std_dA):
    """
    Process latitude and longitude for a given dataset and return amplitude, theta, and phase.

    Parameters:
    - data (xr.Dataset): Full input dataset.
    - lat_lon (tuple): Tuple containing latitude and longitude values.   

    Returns:
    - dict: Dictionary containing amplitude, theta, and phase along with their coordinates.
    """
    
    lat, lon = lat_lon
    ds = compute_amplitude_and_phase(data.sel(lat=lat, lon=lon), stdA, std_dA)
    result = {}

    var_names = ['amplitude', 'theta', 'phase']
    for var_name in var_names:
        # Select the variable as an xarray DataArray
        var = ds[var_name]
        # Store the variable and its coordinates in a dictionary
        result[var_name] = (var.dims, var.values, {'lat': lat, 'lon': lon})
    return result

def full_dataset(data, stdA, std_dA):
    """
    Process the full dataset in parallel for amplitude, theta, and phase.

    Parameters:
    - data (xr.Dataset): Input dataset containing spatial and temporal data.
    - stdA: standard deviation of A over all times, lons and lats
    - std_dA: standard deviation of dA/dt over all times, lons and lats 

    Returns:
    - xr.Dataset: Dataset containing amplitude, theta, and phase for each latitude and longitude combination.
    """
    # Define the lat-lon combinations to process
    lat_lons = [(lat, lon) for lat in data.lat for lon in data.lon]

    # Create a multiprocessing pool with one process per CPU core
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    # Call the process_lat_lon function for each lat-lon combination in parallel
    results = pool.starmap(process_lat_lon, [(data, lat_lon, stdA, std_dA) for lat_lon in lat_lons])

    # Close the pool to free up resources
    pool.close()

    # Merge the results into a single Dataset
    ds_concatenated = xr.Dataset(
        {
            'amplitude': (['time', 'lat', 'lon'], np.nan * np.zeros((len(data.time), len(data.lat), len(data.lon)))),
            'theta': (['time', 'lat', 'lon'], np.nan * np.zeros((len(data.time), len(data.lat), len(data.lon)))),
            'phase': (['time', 'lat', 'lon'], np.nan * np.zeros((len(data.time), len(data.lat), len(data.lon))))
        },
        coords={
            'time': data.time,
            'lat': data.lat,
            'lon': data.lon
        }
    )

    # Assign the computed values to the appropriate locations in the Dataset
    var_names = ['amplitude', 'theta', 'phase']
    for result in results:
        for var_name in var_names:
            dims, values, coords = result[var_name]
            ds_concatenated[var_name].loc[coords] = xr.DataArray(values, dims=dims, coords=coords)

    return ds_concatenated
