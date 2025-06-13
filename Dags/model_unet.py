# =============================
# File: model_unet.py
# Description: Optimized lightweight U-Net for fast segmentation
# =============================

import tensorflow as tf
from tensorflow.keras import layers, models

def build_light_unet(input_shape=(64, 64, 1)):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = layers.SeparableConv2D(8, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.SeparableConv2D(16, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(p2)

    # Decoder
    u4 = layers.UpSampling2D()(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.SeparableConv2D(16, 3, activation='relu', padding='same')(u4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.SeparableConv2D(8, 3, activation='relu', padding='same')(u5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    model = models.Model(inputs, outputs)
    return model

def main():
    model = build_light_unet()
    model.summary()

if __name__ == '__main__':
    main()
