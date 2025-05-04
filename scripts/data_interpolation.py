import numpy as np

def fill_nan_with_mean_per_sample_channel(x : np.ndarray) -> np.ndarray:
    for i in range(x.shape[0]):  
        for j in range(x.shape[3]):  
            slice_2d = x[i, :, :, j]
            mean_val = np.nanmean(slice_2d)  
            mask_nan = np.isnan(slice_2d)
            slice_2d[mask_nan] = mean_val
            x[i, :, :, j] = slice_2d
    return x