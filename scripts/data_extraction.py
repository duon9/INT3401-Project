import numpy as np 
import pandas as pd 
from typing import Optional, List, Dict, Union, Literal
import xarray as xr
import os

def extract_meteorological_data(
    ds : xr.Dataset,
    var_name : str,
    pressure_level : Union[int, float],
) -> np.ndarray:

    matrix = ds[var_name].sel(
        isobaricInhPa=pressure_level,
        method = 'nearest'
    ).values

    return matrix

def create_multichannel_matrix(
    ds : xr.Dataset,
    var_pressure_dict : Dict[str, Union[int, float]]
) -> np.ndarray:

    layers = [
        extract_meteorological_data(ds, var_name, pressure_level)
        for var_name, pressure_level in var_pressure_dict.items()
    ]

    multichannel_matrix = np.stack(layers, axis=-1)

    return multichannel_matrix

def create_batch_from_folder(
    folder_path: str,
    var_pressure_dict: Dict[str, Union[int, float]]
) -> np.ndarray:

    batch_images = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.nc'):
            file_path = os.path.join(folder_path, file_name)
            ds = xr.open_dataset(file_path)
            multichannel_matrix = create_multichannel_matrix(ds, var_pressure_dict)
            batch_images.append(multichannel_matrix)
            ds.close()

    batch_array = np.stack(batch_images)
    
    return batch_array