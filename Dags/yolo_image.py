import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

def save_image_from_dicom(dicom_dir, output_img_dir, patient_id):
    """Convert a DICOM file to a 3-channel JPG image and save it."""
    img_path = os.path.join(output_img_dir, f"{patient_id}.jpg")
    if os.path.exists(img_path):
        return

    dicom_path = os.path.join(dicom_dir, f"{patient_id}.dcm")
    dicom_data = pydicom.dcmread(dicom_path)
    img_array = dicom_data.pixel_array
    img_rgb = np.stack([img_array] * 3, axis=-1)
    cv2.imwrite(img_path, img_rgb)

def save_yolo_label(label_dir, patient_id, row=None):
    """Save bounding box annotation in YOLO format."""
    img_size = 1024
    label_path = os.path.join(label_dir, f"{patient_id}.txt")

    with open(label_path, "a") as f:
        if row is None:
            return
        x, y, w, h = row[1], row[2], row[3], row[4]
        cx = (x + w / 2) / img_size
        cy = (y + h / 2) / img_size
        nw = w / img_size
        nh = h / img_size
        f.write(f"0 {cx} {cy} {nw} {nh}\n")

def prepare_yolo_dataset(dicom_dir, output_img_dir, label_dir, annotations_df):
    """Process dataset and save images and labels in YOLO format."""
    for row in tqdm(annotations_df.values):
        patient_id = row[0]
        target = row[5]
        img_path = os.path.join(output_img_dir, f"{patient_id}.jpg")
        if target == 0:
            continue
        save_yolo_label(label_dir, patient_id, row)
        if not os.path.exists(img_path):
            save_image_from_dicom(dicom_dir, output_img_dir, patient_id)

def main():
    TRAIN_IMG_DIR = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_images"
    LABEL_CSV = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_labels.csv"
    OUTPUT_IMG_DIR = "yolo_images"
    OUTPUT_LABEL_DIR = "yolo_labels"

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    labels_df = pd.read_csv(LABEL_CSV)
    prepare_yolo_dataset(TRAIN_IMG_DIR, OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR, labels_df)

if __name__ == "__main__":
    main()
