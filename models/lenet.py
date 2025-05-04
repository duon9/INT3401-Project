import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras

def LeNet(input_shape : tuple = (33, 33, 5)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(6, 
                      kernel_size=(5, 5),
                      activation='tanh', 
                      padding='same'),
        
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Conv2D(16, 
                      kernel_size=(5, 5), 
                      activation='tanh'),
        
        layers.AveragePooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),

        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(1, activation='sigmoid') 
    ])
    
    return model