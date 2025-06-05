#Data Preprocessing for both image and .csv files 

# Data manipulation and analysis
import pandas as pd         # For working with DataFrames (structured data)
import numpy as np          # For numerical operations and arrays

# Visualization
import matplotlib.pyplot as plt  # For general plotting
import seaborn as sns            # For enhanced statistical visualizations

# DICOM image processing
import pydicom                # For reading and handling DICOM medical images

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
