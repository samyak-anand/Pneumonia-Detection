import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# --- Configuration ---
IMAGE_SIZE = (64, 64)
NPY_DIR = 'npy_data'
os.makedirs(NPY_DIR, exist_ok=True)

# --- Get DICOM Paths ---
def get_dicom_paths(directory_path, label=None, verbose=True):
    dicom_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.lower().endswith('.dcm')
    ]
    if verbose:
        label_text = f" in the '{label}' directory" if label else ""
        print(f" Found {len(dicom_paths)} DICOM files{label_text}.")
    return dicom_paths

TRAIN_IMG_DIR = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_images"
TEST_IMG_DIR  = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_test_images"

train_img_path = get_dicom_paths(TRAIN_IMG_DIR, label="train")
test_img_path  = get_dicom_paths(TEST_IMG_DIR, label="test")

# --- Functions ---
def decode_image(file_path):
    image = pydicom.dcmread(file_path).pixel_array
    image = cv2.resize(image, IMAGE_SIZE)
    return image / 255.0

def parse_metadata(dcm):
    unpacked = {}
    for _ in dcm:
        pass
    for tag, elem in dcm.items():
        unpacked[elem.keyword] = elem.value
    return unpacked

def create_labels(df):
    if 'SeriesDescription' not in df.columns:
        raise ValueError("Missing 'SeriesDescription' in metadata.")
    return (df['SeriesDescription'] == 'view: PA').astype(int).values

def prepare_images(paths):
    images = [decode_image(p) for p in tqdm(paths)]
    gray_stack = np.array(images)
    return np.repeat(gray_stack[..., np.newaxis], 3, -1)

# --- Load CSV Labels and Class Weights ---
labels = pd.read_csv(
    r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_labels.csv"
)

count_normal = (labels['Target'] == 0).sum()
count_pneumonia = (labels['Target'] == 1).sum()
train_count = len(train_img_path)

weights = {
    0: (1 / count_normal) * (train_count / 2.0),
    1: (1 / count_pneumonia) * (train_count / 2.0)
}
print("Class Weights:", weights)

# --- Read and Parse Metadata ---
train_dcms = [pydicom.dcmread(p) for p in tqdm(train_img_path)]
test_dcms = [pydicom.dcmread(p) for p in tqdm(test_img_path)]

train_meta = [parse_metadata(dcm) for dcm in tqdm(train_dcms)]
test_meta = [parse_metadata(dcm) for dcm in tqdm(test_dcms)]

train_df = pd.DataFrame(train_meta)
test_df = pd.DataFrame(test_meta)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

train_Y = create_labels(train_df)
test_Y = create_labels(test_df)


def create_segmentation_mask(img_id, boxes_df, orig_size=(1024, 1024), new_size=(64, 64)):
    mask = np.zeros(new_size, dtype=np.uint8)
    boxes = boxes_df[boxes_df['patientId'] == img_id]

    scale_x = new_size[0] / orig_size[0]
    scale_y = new_size[1] / orig_size[1]

    for _, row in boxes.iterrows():
        x1 = int(row['x'] * scale_x)
        y1 = int(row['y'] * scale_y)
        x2 = int((row['x'] + row['width']) * scale_x)
        y2 = int((row['y'] + row['height']) * scale_y)

        x1, x2 = np.clip([x1, x2], 0, new_size[0])
        y1, y2 = np.clip([y1, y2], 0, new_size[1])

        mask[y1:y2, x1:x2] = 1

    return mask
bbox_df = pd.read_csv(r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_labels.csv")
#bbox_df = bbox_df[bbox_df['Target'] == 1].copy()

train_masks = []

for path in tqdm(train_img_path, desc="Generating segmentation masks"):
    img_id = os.path.splitext(os.path.basename(path))[0]
    dcm = pydicom.dcmread(path)
    orig_shape = dcm.pixel_array.shape
    mask = create_segmentation_mask(img_id, bbox_df, orig_size=orig_shape, new_size=IMAGE_SIZE)
    train_masks.append(mask[..., np.newaxis])  # Add channel dimension

train_masks = np.array(train_masks, dtype=np.uint8)
np.save(os.path.join(NPY_DIR, 'train_masks.npy'), train_masks)

# --- Image Preprocessing ---
train_X_rgb = prepare_images(train_img_path)
test_X_rgb = prepare_images(test_img_path)

test_masks = []

for path in tqdm(test_img_path, desc="Generating test segmentation masks"):
    img_id = os.path.splitext(os.path.basename(path))[0]
    dcm = pydicom.dcmread(path)
    orig_shape = dcm.pixel_array.shape
    mask = create_segmentation_mask(img_id, bbox_df, orig_size=orig_shape, new_size=IMAGE_SIZE)
    test_masks.append(mask[..., np.newaxis])  # Add channel dimension

test_masks = np.array(test_masks, dtype=np.uint8)
np.save(os.path.join(NPY_DIR, 'test_masks.npy'), test_masks)

# --- Save Processed Data ---
np.save(os.path.join(NPY_DIR, 'train_X_rgb.npy'), train_X_rgb)
np.save(os.path.join(NPY_DIR, 'test_X_rgb.npy'), test_X_rgb)
np.save(os.path.join(NPY_DIR, 'train_Y.npy'), train_Y)
np.save(os.path.join(NPY_DIR, 'test_Y.npy'), test_Y)

print(f"\n All processed arrays saved to: {NPY_DIR}")