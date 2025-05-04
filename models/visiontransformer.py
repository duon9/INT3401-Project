import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras

class VisionTransformer:
    def __init__(
        self,
        input_shape : tuple = (33, 33, 5),
        patch_size : tuple =(16, 16),
        projection_dim : int=64,
        transformer_layers : int =8,
        num_heads : int=4,
        transformer_units : list[int]=[128, 64],
        mlp_head_units : list[int]=[2048, 1024],
        data_augmentation : bool=None,
        multihead_attention_lsa : bool=None,
        diag_attn_mask : bool=None,
        vanilla=True
    ) -> None:
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.mlp_head_units = mlp_head_units
        self.data_augmentation = data_augmentation
        self.multihead_attention_lsa = multihead_attention_lsa
        self.diag_attn_mask = diag_attn_mask
        self.vanilla = vanilla

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        if self.data_augmentation:
            x = self.data_augmentation(inputs)
        else:
            x = inputs

        patch_height, patch_width = self.patch_size
        num_patches = (self.input_shape[0] // patch_height) * (self.input_shape[1] // patch_width)
        x = layers.Conv2D(
            filters=self.projection_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid"
        )(x)
        x = layers.Reshape((num_patches, self.projection_dim))(x)

        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=self.projection_dim)(positions)
        encoded_patches = x + pos_embedding

        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

            if not self.vanilla and self.multihead_attention_lsa:
                attention_output = self.multihead_attention_lsa(
                    num_heads=self.num_heads,
                    key_dim=self.projection_dim,
                    dropout=0.1
                )(x1, x1, attention_mask=self.diag_attn_mask)
            else:
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.projection_dim,
                    dropout=0.1
                )(x1, x1)

            x2 = layers.Add()([attention_output, encoded_patches])

            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)

            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        logits = layers.Dense(1, activation = 'sigmoid')(features)

        model = keras.Model(inputs=inputs, outputs=logits)
        return model