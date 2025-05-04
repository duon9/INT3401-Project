import numpy as np

def z_score_normalization(x: np.ndarray) -> np.ndarray:
    """
    Normalize the input data using z-score normalization.
    
    Parameters:
    x (np.ndarray): Input data to be normalized.
    
    Returns:
    np.ndarray: Normalized data.
    """
    mean = np.nanmean(x, axis=(0, 1, 2), keepdims=True)
    std = np.nanstd(x, axis=(0, 1, 2), keepdims=True)
    
    # Avoid division by zero
    std[std == 0] = 1e-10
    
    return (x - mean) / std

def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """
    Normalize the input data using min-max normalization.
    
    Parameters:
    x (np.ndarray): Input data to be normalized.
    
    Returns:
    np.ndarray: Normalized data.
    """
    min_val = np.nanmin(x, axis=(0, 1, 2), keepdims=True)
    max_val = np.nanmax(x, axis=(0, 1, 2), keepdims=True)
    
    # Avoid division by zero
    max_val[max_val == min_val] = min_val[max_val == min_val] + 1e-10
    
    return (x - min_val) / (max_val - min_val)