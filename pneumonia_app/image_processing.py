import streamlit as st
import pandas as pd
import os
import numpy as np
import pydicom
import cv2
from PIL import Image
from pydicom.multival import MultiValue
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt

# --- Function to Extract Patient Metadata from DICOM ---
def extract_patient_info(datasets):
    patient_info_list = []
    for ds in datasets:
        pixel_spacing = ds.get("PixelSpacing", ["Unknown", "Unknown"])
        if isinstance(pixel_spacing, (list, tuple, MultiValue)) and len(pixel_spacing) == 2:
            pixel_spacing_y = pixel_spacing[0]
            pixel_spacing_x = pixel_spacing[1]
        else:
            pixel_spacing_y = pixel_spacing_x = "Unknown"

        patient_info = {
            "Patient ID": str(ds.get("PatientID", "Unknown")),
            "Patient Name": str(ds.get("PatientName", "Unknown")),
            "Patient Sex": str(ds.get("PatientSex", "Unknown")),
            "Patient Age": str(ds.get("PatientAge", "Unknown")),
            "Patient Birth Date": str(ds.get("PatientBirthDate", "Unknown")),
            "Modality": str(ds.get("Modality", "Unknown")),
            "Body Part Examined": str(ds.get("BodyPartExamined", "Unknown")),
            "View Position": str(ds.get("ViewPosition", "Unknown")),
            "Rows": str(ds.get("Rows", "Unknown")),
            "Columns": str(ds.get("Columns", "Unknown")),
            "Pixel Spacing X": str(pixel_spacing_x),
            "Pixel Spacing Y": str(pixel_spacing_y),
            "Photometric Interpretation": str(ds.get("PhotometricInterpretation", "Unknown")),
            "Bits Allocated": str(ds.get("BitsAllocated", "Unknown")),
            "Bits Stored": str(ds.get("BitsStored", "Unknown")),
            "High Bit": str(ds.get("HighBit", "Unknown")),
            "Study Date": str(ds.get("StudyDate", "Unknown")),
            "Study Time": str(ds.get("StudyTime", "Unknown")),
            "Series Description": str(ds.get("SeriesDescription", "Unknown")),
            "SOP Instance UID": str(ds.get("SOPInstanceUID", "Unknown")),
            "Conversion Type": str(ds.get("ConversionType", "Unknown")),
            "Lossy Image Compression": str(ds.get("LossyImageCompression", "Unknown")),
            "Lossy Compression Method": str(ds.get("LossyImageCompressionMethod", "Unknown")),
            "Transfer Syntax UID": str(ds.file_meta.get("TransferSyntaxUID", "Unknown")),
        }
        patient_info_list.append(patient_info)
    return patient_info_list

# --- Main Viewer Function ---
def run_image_viewer():
    st.subheader("📂 Upload Datasets for Image Visualization")

    details_file = st.file_uploader("📝 Upload `stage_2_detailed_class_info.csv`", type="csv")
    labels_file = st.file_uploader("📝 Upload `stage_2_train_labels.csv`", type="csv")
    dicom_dir = st.text_input("📁 Enter path to DICOM folder", help="E.g., C:/data/dicom_images")

    if not (details_file and labels_file and dicom_dir):
        st.info("Please upload both CSV files and provide the DICOM folder path.")
        return

    try:
        details_df = pd.read_csv(details_file)
        labels_df = pd.read_csv(labels_file)
    except Exception as e:
        st.error(f"❌ Failed to load CSVs: {e}")
        return

    details_df = details_df.drop_duplicates('patientId').reset_index(drop=True)
    full_df = pd.merge(labels_df, details_df, on='patientId', how='left')
    full_df.fillna(0, inplace=True)

    st.success("✅ Files processed. Pick a patient below to view the image and metadata.")

    patient_ids = full_df["patientId"].unique()
    selected_patient = st.selectbox("🔍 Select a Patient ID", patient_ids)

    dcm_path = os.path.join(dicom_dir, f"{selected_patient}.dcm")
    if not os.path.exists(dcm_path):
        st.error(f"❌ DICOM not found: {dcm_path}")
        return

    try:
        dcm = pydicom.dcmread(dcm_path)
        image = dcm.pixel_array
    except Exception as e:
        st.error(f"❌ Failed to load DICOM image: {e}")
        return

    image_size = 1024
    target_size = 128
    factor = target_size / image_size
    img_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    img_rgb = Image.fromarray(img_resized).convert("RGB")
    img_np = preprocess_input(np.array(img_rgb, dtype=np.float32))

    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    img_with_boxes = cv2.cvtColor(img_resized.copy(), cv2.COLOR_GRAY2BGR)
    patient_rows = full_df[full_df["patientId"] == selected_patient]

    for _, row in patient_rows.iterrows():
        if row["Target"] == 1:
            x1 = int(row["x"] * factor)
            y1 = int(row["y"] * factor)
            x2 = int((row["x"] + row["width"]) * factor)
            y2 = int((row["y"] + row["height"]) * factor)
            mask[y1:y2, x1:x2] = 1
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Original Image")
        st.image(np.clip((img_np + [123.68, 116.779, 103.939]), 0, 255).astype(np.uint8), use_column_width=True)

    with col2:
        st.markdown("### 🔴 Image with Red Boxes")
        st.image(img_with_boxes, caption="Image with Red Bounding Boxes", use_column_width=True)

    # --- Matplotlib Visualization Overlay ---
    if np.any(mask):  # Only if at least one region is present
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_resized, cmap='gray')
        ax.imshow(mask, cmap='Reds', alpha=0.3)

        for _, row in patient_rows.iterrows():
            if row["Target"] == 1:
                x1 = int(row["x"] * factor)
                y1 = int(row["y"] * factor)
                width = int(row["width"] * factor)
                height = int(row["height"] * factor)
                rect = plt.Rectangle((x1, y1), width, height, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)

        ax.set_title("Pneumonia Region Overlay")
        ax.axis('off')
        st.markdown("### 🧊 Matplotlib Overlay")
        st.pyplot(fig)
    else:
        st.info("No pneumonia region detected to overlay.")

    st.markdown("### 💡 Additional Information")
    st.write(f"Patient ID: {selected_patient}")
    st.write(f"Mask size: {mask.shape[0]} x {mask.shape[1]}")
    st.write(f"Image size: {img_resized.shape[0]} x {img_resized.shape[1]}")

    # --- Diagnosis Class ---
    patient_class = patient_rows["class"].iloc[0] if "class" in patient_rows.columns else "Unknown"
    st.markdown("### 🧾 Diagnosis Class")
    st.write(f"**Class Label:** {patient_class}")

    # --- Display Metadata ---
    st.markdown("### 🧬 Patient Metadata from DICOM")
    metadata = extract_patient_info([dcm])
    if metadata:
        meta = metadata[0]

        with st.expander("👤 Patient Information"):
            st.write(f"**Patient ID:** {meta['Patient ID']}")
            st.write(f"**Name:** {meta['Patient Name']}")
            st.write(f"**Sex:** {meta['Patient Sex']}")
            st.write(f"**Age:** {meta['Patient Age']}")
            st.write(f"**Birth Date:** {meta['Patient Birth Date']}")

        with st.expander("🩺 Study & Series Info"):
            st.write(f"**Study Date:** {meta['Study Date']}")
            st.write(f"**Study Time:** {meta['Study Time']}")
            st.write(f"**Series Description:** {meta['Series Description']}")
            st.write(f"**SOP Instance UID:** {meta['SOP Instance UID']}")

        with st.expander("🖼️ Image Acquisition"):
            st.write(f"**Modality:** {meta['Modality']}")
            st.write(f"**Body Part Examined:** {meta['Body Part Examined']}")
            st.write(f"**View Position:** {meta['View Position']}")
            st.write(f"**Resolution:** {meta['Rows']} x {meta['Columns']}")
            st.write(f"**Pixel Spacing:** {meta['Pixel Spacing X']} x {meta['Pixel Spacing Y']}")
            st.write(f"**Photometric Interpretation:** {meta['Photometric Interpretation']}")
            st.write(f"**Bits Allocated / Stored / High Bit:** {meta['Bits Allocated']} / {meta['Bits Stored']} / {meta['High Bit']}")

        with st.expander("📦 Compression & Transfer"):
            st.write(f"**Conversion Type:** {meta['Conversion Type']}")
            st.write(f"**Lossy Compression:** {meta['Lossy Image Compression']}")
            st.write(f"**Compression Method:** {meta['Lossy Compression Method']}")
            st.write(f"**Transfer Syntax UID:** {meta['Transfer Syntax UID']}")
    else:
        st.warning("No metadata found for this patient.")

# --- Run the viewer ---
if __name__ == "__main__":
    run_image_viewer()
