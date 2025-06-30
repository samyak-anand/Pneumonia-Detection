# =============================
# File: model_cnn.py
# Description: Defines and compiles CNN models
# =============================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization

def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_fcnn_model():
    model = Sequential([
        Flatten(input_shape=(128, 128, 3)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_deep_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        Conv2D(64, 3, activation='relu'),
        Conv2D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.4),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def main():
    print("Model definitions loaded. Use build_custom_cnn(), build_fcnn_model(), or build_deep_cnn().")

if __name__ == '__main__':
    main()