#Data Preprocessing for both image and .csv files 

# Data manipulation and analysis
import pandas as pd         # For working with DataFrames (structured data)
import numpy as np          # For numerical operations and arrays

# Visualization
import matplotlib.pyplot as plt  # For general plotting
import seaborn as sns            # For enhanced statistical visualizations

# DICOM image processing
import pydicom                # For reading and handling DICOM medical images
from pydicom.multival import MultiValue
# Image manipulation
from PIL import Image         # For general-purpose image processing (e.g., resizing, saving)

# Display tools (useful in Jupyter Notebooks)
from IPython.display import HTML, display  # For rendering HTML and interactive elements
import os


def get_dicom_paths(directory_path, label=""):
    """
    Scans a given directory for DICOM (.dcm) files, returns their full paths,
    and displays the count of images found.

    Parameters:
    - directory_path (str): Path to the directory containing DICOM files.
    - label (str): Optional label for the directory (e.g., 'train', 'test') for display purposes.

    Returns:
    - List of full paths to DICOM files.
    """
    # Collect all .dcm file paths in the directory
    dicom_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith('.dcm')
    ]
    
    # Display the count using HTML formatting
    display(HTML(f"<strong>Number of images present in the {label} directory is {len(dicom_paths)}</strong>"))
    
    return dicom_paths


TRAIN_IMG_DIR = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_images"
TEST_IMG_DIR  = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_test_images"

# Get DICOM file paths
train_img_path = get_dicom_paths(TRAIN_IMG_DIR, label="train")
test_img_path  = get_dicom_paths(TEST_IMG_DIR, label="test")


# Load detailed class information per patient
# This CSV contains columns like patientId and class (e.g., 'Lung Opacity', 'No Finding', etc.)
df_detail_class = pd.read_csv(r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_detailed_class_info.csv")

# Load training labels
# This CSV includes patientId, target (1 = pneumonia, 0 = no pneumonia), and bounding box coordinates
df_label = pd.read_csv(r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_labels.csv")


def display_dicom_samples(image_paths, num_samples=5, title="DICOM Samples"):
    """
    Displays a specified number of DICOM images from a given list of paths.

    Parameters:
    - image_paths (list): List of file paths to DICOM images.
    - num_samples (int): Number of images to display.
    - title (str): Title prefix for each image displayed.
    """
    for idx, img_path in enumerate(image_paths[:num_samples]):
        try:
            dicom_img = pydicom.dcmread(img_path)

            if hasattr(dicom_img, 'pixel_array'):
                plt.figure(figsize=(6, 6))
                plt.imshow(dicom_img.pixel_array, cmap='gray')
                plt.title(f"{title} - {os.path.basename(img_path)}")
                plt.axis('off')
                plt.show()
            else:
                print(f"No pixel data in: {img_path}")

        except Exception as e:
            print(f"Error reading {img_path}: {e}")




def df_overview(df):
    """
    Display an overview of the DataFrame including:
    - Null values
    - Shape
    - Data types
    - Duplicate records
    - Unique value counts
    - Column names
    - Summary info
    """
    
    print('========== DataFrame Overview ==========\n')

    # Show total missing values per column
    print("Null values per column:\n", df.isnull().sum())
    print('\n-----------------------------------------')

    # Show number of rows and columns
    print(f"Shape of DataFrame: {df.shape}")
    print('\n-----------------------------------------')

    # Display basic information about DataFrame (memory usage, types, non-null counts)
    print("DataFrame Info:")
    df.info()
    print('\n-----------------------------------------')

    # Show data types of each column
    print("Data types:\n", df.dtypes)
    print('\n-----------------------------------------')

    # Show column names
    print("Column names:\n", df.columns.tolist())
    print('\n-----------------------------------------')

    # Check for duplicate rows
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print('\n-----------------------------------------')

    # Count of unique values per column
    print("Unique value counts:\n", df.nunique())
    print('\n=========================================')

def clean_patient_info(patient_info_list, dedup_column='patientId'):
    """
    Cleans patient info by dropping duplicates.

    Parameters:
        patient_info_list (list): List of patient dictionaries.
        dedup_column (str or list): Column(s) to drop duplicates by (default is 'patientId').

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    df = pd.DataFrame(patient_info_list)

    # Drop duplicates by column (e.g., 'patientId' or ['pixelSpacing_X', 'pixelSpacing_Y'])
    df = df.drop_duplicates(subset=dedup_column, keep='first')

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

def df_describe(df):
    """
    Display descriptive statistics for numerical and categorical columns in the DataFrame.
    """
    print('========== DataFrame Description ==========\n')

    # Numeric columns
    print("Numerical Columns Description:\n")
    print(df.describe())

    # Categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print('\nCategorical Columns Description:\n')
        print(df[cat_cols].describe())
    else:
        print('\nNo categorical columns to describe.')

    print('\n=============================================')

def missing_data(df):
    """
    Returns a DataFrame showing the total and percentage of missing values per column,
    sorted by the most missing data first.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A summary of missing data with columns ['Total', 'Percent'].
    """
    # Total missing values per column
    total = df.isnull().sum().sort_values(ascending=False)

    # Percentage of missing values
    percent = (total / len(df) * 100).sort_values(ascending=False)

    # Combine results into one DataFrame
    missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_df

def get_feature_distribution(data, feature):
    # Get the count for each label
    label_counts = data[feature].value_counts()

    # Get total number of samples
    total_samples = len(data)

    # Count the number of items in each class
    print("Feature: {}".format(feature))
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        count = label_counts.values[i]
        percent = int((count / total_samples) * 10000) / 100
        print("{:<30s}:   {} , {}%".format(label, count, percent))


def read_dicom_files(file_paths):
    """
    Reads multiple DICOM files and returns them as a list of DICOM datasets.

    Parameters:
    - file_paths (list): List of full paths to .dcm files.

    Returns:
    - List of pydicom Dataset objects.
    """
    dicom_data_list = []

    for path in file_paths:
        try:
            # Load the DICOM file into a pydicom Dataset
            dicom_data = pydicom.dcmread(path)
            dicom_data_list.append(dicom_data)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return dicom_data_list




def extract_patient_info(datasets):
    patient_info_list = []

    for ds in datasets:
        pixel_spacing = ds.get("PixelSpacing", ["Unknown", "Unknown"])
        #print("PixelSpacing raw value:", pixel_spacing)  # Debug print

        if isinstance(pixel_spacing, (list, tuple, MultiValue)) and len(pixel_spacing) == 2:
            pixel_spacing_y = pixel_spacing[0]
            pixel_spacing_x = pixel_spacing[1]
        else:
            pixel_spacing_y = pixel_spacing_x = "Unknown"

        patient_info = {
            "patientId": str(ds.get("PatientID", "Unknown")),
            "sex": str(ds.get("PatientSex", "Unknown")),
            "age": int(ds.get("PatientAge", "Unknown")),
            "examined_part": str(ds.get("BodyPartExamined", "Unknown")),
            "rows": int(ds.get("Rows", "Unknown")),
            "columns": str(ds.get("Columns", "Unknown")),
            "viewPosition": str(ds.get("ViewPosition", "Unknown")),
            "pixelSpacing_X": float(pixel_spacing_x),
            "pixelSpacing_Y": float(pixel_spacing_y),
            "modality": str(ds.get("Modality", "Unknown"))
        }

        patient_info_list.append(patient_info)

    return patient_info_list
