import numpy as np
import scipy.ndimage as ndi
from typing import Union

def random_shear(data : np.ndarray, 
                 shear_factor : float=0.2) -> np.ndarray:
    
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    
    return ndi.affine_transform(data, 
                                shear_matrix, 
                                order=1, 
                                mode='nearest')

def random_translate(data : np.ndarray, 
                     max_translation : int=5) -> np.ndarray:
    
    translation = np.random.randint(-max_translation, 
                                    max_translation, 
                                    size=2)
    
    return ndi.shift(data, 
                     shift=translation, 
                     order=1, 
                     mode='nearest')

def random_rotate(data : np.ndarray, 
                  max_angle : Union[int, float]=30) -> np.ndarray:
    
    angle = np.random.uniform(-max_angle, 
                              max_angle)  
    
    return ndi.rotate(data, 
                      angle, 
                      reshape=False, 
                      order=1, 
                      mode='nearest')


def add_gaussian_noise(data : np.ndarray, mean : float=0.0, std : float=0.1) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=std, size=data.shape)
    return data + noise
