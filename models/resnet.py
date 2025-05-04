import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras


def residual_block(
    x, 
    filters, 
    downsample=False, 
    kernel_size=(3, 3), 
    kernel_initializer='he_normal', 
    use_batch_norm=True, 
    use_dropout=False, 
    dropout_rate=0.2, 
    activation='relu'
):
    shortcut = x
    strides = (2, 2) if downsample else (1, 1)

    if downsample: 
        shortcut = layers.Conv2D(
            filters, 
            (1, 1), 
            strides=strides, 
            padding='same', 
            kernel_initializer=kernel_initializer, 
            use_bias=not use_batch_norm
        )(shortcut)
        if use_batch_norm:
            shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(
        filters, 
        kernel_size, 
        strides=strides, 
        padding='same', 
        kernel_initializer=kernel_initializer, 
        use_bias=not use_batch_norm
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if use_dropout:
        x = layers.Dropout(dropout_rate, seed = 42)(x)

    x = layers.Conv2D(
        filters, 
        kernel_size, 
        strides=(1, 1), 
        padding='same', 
        kernel_initializer=kernel_initializer, 
        use_bias=not use_batch_norm
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    
    return x

def build_deep_resnet(
    input_shape=(33, 33, 9), 
    block_structure=[(64, 2), (128, 2), (256,2)], 
    use_batch_norm=True, 
    use_dropout=False, 
    dropout_rate=0.1, 
    kernel_initializer='he_normal'
):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        block_structure[0][0], 
        (2, 2), 
        strides=(1, 1), 
        padding='same', 
        kernel_initializer=kernel_initializer, 
        use_bias=not use_batch_norm
    )(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    for i, (filters, num_blocks) in enumerate(block_structure):
        for j in range(num_blocks):
            x = residual_block(
                x, 
                filters=filters, 
                downsample=(j == 0 and i > 0),  
                kernel_initializer=kernel_initializer,
                use_batch_norm=use_batch_norm,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate
            )

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dropout(0.1, seed = 42)(x)
    # x = layers.Dense(128, activation = 'relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model