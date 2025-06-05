# ==========================
# Data Preprocessing for DICOM Images and CSV Metadata
# ==========================

# --- Import Required Libraries ---
# Data manipulation
import pandas as pd         # For structured data manipulation
import numpy as np          # For numerical computations

# Visualization
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns            # For statistical visualizations

# DICOM handling
import pydicom                # To read DICOM image files
from pydicom.multival import MultiValue  # Handles multi-valued DICOM fields

# Image handling
from PIL import Image         # For resizing/saving images

# Display tools
from IPython.display import HTML, display  # For HTML rendering in notebooks
import os  # For interacting with the file system
# ============================================================================


# --- Function to Get All DICOM File Paths in a Directory ---
def get_dicom_paths(directory_path, label=""):
    """
    Scans a directory for DICOM (.dcm) files and returns their paths.
    Displays count for overview.
    
    Parameters:
    - directory_path (str): Directory to scan for .dcm files
    - label (str): Optional label (e.g. 'train', 'test')

    Returns:
    - List of paths to DICOM files
    """
    dicom_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith('.dcm')
    ]
    display(HTML(f"<strong>Number of images present in the {label} directory is {len(dicom_paths)}</strong>"))
    return dicom_paths

#==========================================================================

# --- Define Image Directories ---
TRAIN_IMG_DIR = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_images"
TEST_IMG_DIR  = r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_test_images"

train_img_path = get_dicom_paths(TRAIN_IMG_DIR, label="train")
test_img_path  = get_dicom_paths(TEST_IMG_DIR, label="test")

"""
        Dislaying the number  of iages present in the folder

"""
print('\n Number of images present in the train directory is:',len(train_img_path))
print('\n Number of images present in the test directory is:',len(test_img_path))

# --- Load CSV Metadata ---
df_detail_class = pd.read_csv(r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_detailed_class_info.csv")
df_label = pd.read_csv(r"C:\Users\samya\PyCharmProject\Pneumonia-Detection_dataset\data\stage_2_train_labels.csv")


"""
        Dislaying the head of dataset
"""
print('\n The detail class dataframe:\n ', df_detail_class.head())
print('\n The lable dataframe :\n', df_label)

#======================================================================================

# --- Function to Display DICOM Samples ---
def display_dicom_samples(image_paths, num_samples=5, title="DICOM Samples"):
    """
    Displays DICOM images from given paths.
    
    Parameters:
    - image_paths (list): Paths to DICOM images
    - num_samples (int): How many to display
    - title (str): Plot title prefix
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



# Display first 5 training images
display_dicom_samples(train_img_path, num_samples=1, title="Train Image")

# Display first 5 test images
display_dicom_samples(test_img_path, num_samples=1, title="Test Image")

#=========================================================================================


# --- Function for General DataFrame Overview ---
def df_overview(df):
    """
    Displays essential information about a DataFrame including:
    null counts, shape, types, duplicate count, column names, and summary.
    """
    print('========== DataFrame Overview ==========\n')
    print("Null values per column:\n", df.isnull().sum())
    print('\n-----------------------------------------')
    print(f"Shape of DataFrame: {df.shape}")
    print('\n-----------------------------------------')
    print("DataFrame Info:")
    df.info()
    print('\n-----------------------------------------')
    print("Data types:\n", df.dtypes)
    print('\n-----------------------------------------')
    print("Column names:\n", df.columns.tolist())
    print('\n-----------------------------------------')
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print('\n-----------------------------------------')
    print("Unique value counts:\n", df.nunique())
    print('\n=========================================')

#passing the first dataset 
df_overview(df_label)
#passing the second dataset
df_overview(df_detail_class)

#=================================================================================

# --- Function to Describe Numeric and Categorical Features ---
def df_describe(df):
    """
    Describes both numerical and categorical columns.
    """
    print('========== DataFrame Description ==========\n')
    print("Numerical Columns Description:\n")
    print(df.describe())

    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print('\nCategorical Columns Description:\n')
        print(df[cat_cols].describe())
    else:
        print('\nNo categorical columns to describe.')

    print('\n=============================================')


#passing the first dataset 
df_describe(df_label)
#passing the second dataset
df_describe(df_detail_class)

# =======================================================================================

def df_clean(data_list, dedup_column=None):
    """
    Convert a list of dictionaries to a DataFrame and remove duplicates.

    Parameters:
    - data_list (list): List of dictionaries representing data records.
    - dedup_column (str or list, optional): Column(s) to use for identifying duplicates.
      If None, duplicates are removed based on all columns.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data_list)
    
    if dedup_column:
        # Remove duplicates based on specified column(s), keep first occurrence
        df = df.drop_duplicates(subset=dedup_column, keep='first')
    
    # Remove any remaining duplicate rows (based on all columns)
    df = df.drop_duplicates()
    
    # Reset index for clean DataFrame
    df.reset_index(drop=True, inplace=True)
    print('\n The dataset has been clean')
    return df

df_detail_class_clean= df_clean(df_detail_class)
df_lable_clean= df_clean(df_label)
#passing the first dataset 
print('\n The missing data in Frist dataset \n ',df_detail_class_clean.head())
#passing the second dataset
print('\n The missing data in Frist dataset \n ',df_lable_clean.head())

# =======================================================================================

# --- Function to Summarize Missing Data ---
def missing_data(df):
    """
    Returns DataFrame with total and % of missing values per column.

    Parameters:
    - df (pd.DataFrame): DataFrame to check

    Returns:
    - pd.DataFrame: Missing value summary
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (total / len(df) * 100).sort_values(ascending=False)
    missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_df


#passing the first dataset 
print('\n The missing data in Frist dataset \n',missing_data(df_lable_clean))
#passing the second dataset
print('\n The missing data in Second dataset \n',missing_data(df_detail_class_clean))

# =======================================================================================
# --- Function to View Distribution of a Feature ---
def get_feature_distribution(data, feature):
    """
    Displays the distribution count and percentage for a given feature.
    """
    label_counts = data[feature].value_counts()
    total_samples = len(data)
    print("Feature: {}".format(feature))
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        count = label_counts.values[i]
        percent = int((count / total_samples) * 10000) / 100
        print("{:<30s}:   {} , {}%".format(label, count, percent))

# ========================================================================================

print("\n Print the vlaue count of different classes in dataset:\n",df_detail_class_clean['class'].value_counts())
print("\n Print the vlaue count of different Target classes in dataset:\n",df_lable_clean['Target'].value_counts())
# =======================================================================================

print('\n merge the dataset')
pnemonia= pd.merge(df_label,df_detail_class, on='patientId', how='inner')
print('\n The Cobined dataset is:\n',pnemonia.head())

# =======================================================================================



# --- Function to Read DICOM Files into Datasets ---
def read_dicom_files(file_paths):
    """
    Reads multiple DICOM files into Dataset objects.

    Parameters:
    - file_paths (list): Paths to .dcm files

    Returns:
    - List of pydicom Dataset objects
    """
    dicom_data_list = []
    for path in file_paths:
        try:
            dicom_data = pydicom.dcmread(path)
            dicom_data_list.append(dicom_data)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return dicom_data_list

# Convert to pandas Series
train_series = pd.Series(train_img_path)
test_series = pd.Series(test_img_path)

# Concatenate using pandas
combined_series = pd.concat([train_series, test_series], ignore_index=True)


print("Combined paths saved to combined_dicom_paths.csv")

# Read DICOM datasets into memory 
combined_series_dicom_datasets = read_dicom_files(combined_series)
print("\n The sample metadata:\n", combined_series_dicom_datasets[1])
# =======================================================================================
# --- Function to Extract Patient Metadata from DICOM ---
def extract_patient_info(datasets):
    """
    Extracts key patient information and image metadata from DICOM datasets.

    Parameters:
    - datasets (list): List of pydicom Dataset objects

    Returns:
    - List of dictionaries with extracted metadata
    """
    patient_info_list = []
    for ds in datasets:
        pixel_spacing = ds.get("PixelSpacing", ["Unknown", "Unknown"])

        # Handle pixel spacing safely even if it's a MultiValue
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

patient_info_table=extract_patient_info(combined_series_dicom_datasets)
patient_info_table=pd.DataFrame(patient_info_table)
print("\n The patient table ",patient_info_table)
# =======================================================================================
