
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import df_detail_class_clean,df_lable_clean,patient_info_table,combined_series_dicom_datasets



# --- Class Distribution ---
plt.figure(figsize=(8, 4))
sns.countplot(data=df_detail_class_clean, x='class')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df_lable_clean, x='Target')
plt.title('Target Distribution')
plt.tight_layout()
plt.show()

# --- Missing Data Heatmaps ---
plt.figure(figsize=(10, 4))
sns.heatmap(df_detail_class_clean.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data in df_detail_class_clean')
plt.show()

plt.figure(figsize=(10, 4))
sns.heatmap(df_lable_clean.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data in df_lable_clean')
plt.show()

# --- Patient Demographics ---
plt.figure(figsize=(6, 4))
patient_info_table['sex'].value_counts().plot(kind='bar')
plt.title('Sex Distribution')
plt.show()

plt.figure(figsize=(6, 4))
patient_info_table['age'].dropna().astype(int).plot(kind='hist', bins=30)
plt.title('Age Distribution')
plt.show()

# --- Image Metadata ---
plt.figure(figsize=(6, 4))
plt.hist(patient_info_table['rows'], bins=30)
plt.title('Image Height (Rows)')
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(patient_info_table['columns'].astype(int), bins=30)
plt.title('Image Width (Columns)')
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(patient_info_table['pixelSpacing_X'], patient_info_table['pixelSpacing_Y'], alpha=0.5)
plt.xlabel('Pixel Spacing X')
plt.ylabel('Pixel Spacing Y')
plt.title('Pixel Spacing Distribution')
plt.show()

# --- Modality and View Position ---
plt.figure(figsize=(6, 4))
patient_info_table['modality'].value_counts().plot(kind='bar')
plt.title('Modality Distribution')
plt.show()

plt.figure(figsize=(6, 4))
patient_info_table['viewPosition'].value_counts().plot(kind='bar')
plt.title('View Position Distribution')
plt.show()
