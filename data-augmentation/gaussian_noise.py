import numpy as np

def add_gaussian_noise(data : np.ndarray, mean : float=0.0, std : float=0.1) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=std, size=data.shape)
    return data + noise
