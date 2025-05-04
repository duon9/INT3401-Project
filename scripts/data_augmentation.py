import numpy as np
import scipy.ndimage as ndi
from typing import Union, Tuple

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


def add_gaussian_noise(data : np.ndarray,
                       mean : float=0.0,
                       std : float=0.1) -> np.ndarray:
    noise = np.random.normal(loc=mean,
                             scale=std,
                             size=data.shape)
    return data + noise

def random_flip(data: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        data = np.flip(data,
                       axis=0)  # vertical
    if np.random.rand() < 0.5:
        data = np.flip(data,
                       axis=1)  # horizontal
    return data

def rotate_90(data: np.ndarray) -> np.ndarray:
    k = np.random.choice([0, 1, 2, 3])
    return np.rot90(data, k=k, axes=(0, 1))

def augment_image(image: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        image = random_shear(image)
    if np.random.rand() < 0.5:
        image = random_translate(image)
    if np.random.rand() < 0.5:
        image = random_rotate(image)
    if np.random.rand() < 0.5:
        image = add_gaussian_noise(image)
    if np.random.rand() < 0.5:
        image = random_flip(image)
    if np.random.rand() < 0.5:
        image = rotate_90(image)
    return image

def augment_batch(batch: np.ndarray) -> np.ndarray:
    return np.stack([augment_image(img) for img in batch], axis=0)

def augment_and_stack(x_batch: np.ndarray,
                      y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    augmented = augment_batch(x_batch)
    x_stacked = np.concatenate([x_batch, augmented], axis=0)
    y_stacked = np.concatenate([y_batch, y_batch], axis=0)  
    return x_stacked, y_stacked