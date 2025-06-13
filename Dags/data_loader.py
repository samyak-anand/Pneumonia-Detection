# =============================
# File: data_loader.py
# Description: Load and preprocess DICOM data for training and testing
# =============================

import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from tqdm import tqdm

def get_dicom_paths(directory_path, label=None, verbose=True):
    dicom_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.lower().endswith('.dcm')
    ]
    if verbose:
        label_text = f" in the '{label}' directory" if label else ""
        print(f"Found {len(dicom_paths)} DICOM files{label_text}.")
    return dicom_paths

def parse_metadata(dcm):
    unpacked_data = {}
    group_elem_to_keywords = {}
    for _ in dcm:
        pass
    for tag, elem in dcm.items():
        group_elem_to_keywords[(tag.group, tag.elem)] = elem.keyword
        unpacked_data[elem.keyword] = elem.value
    return unpacked_data, group_elem_to_keywords

def create_labels(df):
    if 'SeriesDescription' not in df.columns:
        raise ValueError("Missing 'SeriesDescription' column in DataFrame.")
    return (df['SeriesDescription'] == 'view: PA').astype(int).values

def decode_image(filepath):
    image = pydicom.dcmread(filepath).pixel_array
    image = cv2.resize(image, (64, 64))
    return image / 255.0

def load_dataset(img_paths):
    images = [decode_image(p) for p in tqdm(img_paths)]
    return np.array(images)

def main():
    print("Data loader module. Use this in your training pipeline.")

if __name__ == '__main__':
    main()
