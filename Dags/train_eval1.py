# =============================
# File: train_eval.py
# Description: Loads data, trains CNN model, evaluates performance
# =============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import pydicom
from tqdm import tqdm

from data_loader import get_dicom_paths, parse_metadata, create_labels, load_dataset
from model_cnn import build_custom_cnn
from visualization import plot_confusion, plot_sample_images, print_classification_metrics

TRAIN_IMG_DIR = r"C:\\Users\\samya\\PyCharmProject\\Pneumonia-Detection_dataset\\data\\stage_2_train_images"
TEST_IMG_DIR  = r"C:\\Users\\samya\\PyCharmProject\\Pneumonia-Detection_dataset\\data\\stage_2_test_images"
LABEL_CSV     = r"C:\\Users\\samya\\PyCharmProject\\Pneumonia-Detection_dataset\\data\\stage_2_train_labels.csv"

def exponential_decay(lr0, s):
    def decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return decay_fn

def plot_training_metrics(history):
    fig, ax = plt.subplots(5, 1, figsize=(10, 20))
    ax = ax.ravel()
    for i, met in enumerate(['accuracy', 'precision', 'recall', 'AUC', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title(f'Model {met}')
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.show()

def main():
    train_img_paths = get_dicom_paths(TRAIN_IMG_DIR, label="train")
    test_img_paths  = get_dicom_paths(TEST_IMG_DIR, label="test")
    labels_df       = pd.read_csv(LABEL_CSV)

    count_normal    = len(labels_df[labels_df['Target'] == 0])
    count_pneumonia = len(labels_df[labels_df['Target'] == 1])
    train_count     = len(train_img_paths)
    class_weight    = {
        0: (1 / count_normal) * (train_count / 2.0),
        1: (1 / count_pneumonia) * (train_count / 2.0)
    }

    train_dcms = [pydicom.dcmread(p) for p in tqdm(train_img_paths[:5000])]
    test_dcms  = [pydicom.dcmread(p) for p in tqdm(test_img_paths[:5000])]
    train_meta, _ = zip(*[parse_metadata(dcm) for dcm in tqdm(train_dcms)])
    test_meta,  _ = zip(*[parse_metadata(dcm) for dcm in tqdm(test_dcms)])
    train_df = pd.DataFrame(train_meta)
    test_df  = pd.DataFrame(test_meta)
    train_Y = create_labels(train_df)
    test_Y  = create_labels(test_df)

    train_X = load_dataset(train_img_paths[:5000])
    test_X  = load_dataset(test_img_paths[:5000])
    train_X_rgb = np.repeat(train_X[..., np.newaxis], 3, -1)
    test_X_rgb  = np.repeat(test_X[..., np.newaxis], 3, -1)

    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='AUC')
    ]

    model = build_custom_cnn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(exponential_decay(0.01, 20))
    ]

    history = model.fit(
        train_X_rgb, train_Y,
        epochs=30,
        batch_size=128,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    results = model.evaluate(test_X_rgb, test_Y)
    print("Evaluation Results:", dict(zip(model.metrics_names, results)))

    plot_training_metrics(history)

    predictions = (model.predict(test_X_rgb) > 0.5).astype(int)
    print_classification_metrics(test_Y, predictions)
    plot_confusion(test_Y, predictions)
    plot_sample_images(test_X, test_Y, predictions.flatten(), n=9)

if __name__ == '__main__':
    main()
