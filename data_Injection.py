import os
import zipfile
import subprocess
import pandas as pd

os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\samya\.kaggle'
competition_name = 'rsna-pneumonia-detection-challenge'

print("Downloading dataset from Kaggle...")

result = subprocess.run(
    ['kaggle', 'competitions', 'download', '-c', competition_name], 
    capture_output=True, 
    text=True,
    encoding='utf-8'  # specify UTF-8 explicitly
)

if result.returncode != 0:
    print("Error downloading dataset:", result.stderr)
    exit(1)

print("Download completed!")

# Unzip the downloaded file and continue as before
zip_file = competition_name + '.zip'
extract_folder = os.path.join(os.getcwd(), 'data')

os.makedirs(extract_folder, exist_ok=True)

print("Extracting dataset...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
print(f"Files extracted to: {extract_folder}")

csv_file = os.path.join(extract_folder, 'stage_2_train_labels.csv')

if os.path.isfile(csv_file):
    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    print(df.head())
else:
    print(f"CSV file not found at {csv_file}")
