# 🩺 Pneumonia Detection in Chest X-rays using Deep Learning  
*A solution to the RSNA Pneumonia Detection Challenge*

![RSNA Banner](http://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2016/05/18/13/02/ww5r032t-8col-jpg.jpg)

---

## 📘 Overview

This project aims to develop a deep learning model capable of **detecting pneumonia in chest radiographs (X-rays)** by identifying areas of lung opacity, which are common indicators of pneumonia. The solution leverages labeled DICOM images and annotated bounding boxes from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

---

## ❓ What is Pneumonia?

**Pneumonia** is a serious lung infection that causes **inflammation in the air sacs (alveoli)**. These air sacs may fill with fluid or pus, making it difficult to breathe. Pneumonia can affect one or both lungs and can be caused by:

- Bacteria (e.g., *Streptococcus pneumoniae*)
- Viruses (e.g., influenza, RSV, COVID-19)
- Fungi (less common, in immunocompromised patients)

### 🔬 Radiological Diagnosis

On a **chest X-ray (CXR)**, pneumonia typically appears as **areas of increased opacity** (whiteness) due to fluid or infection in the lung tissue. However, distinguishing pneumonia from other causes of opacity (e.g., tumors, pulmonary edema, pleural effusion, atelectasis) requires experience and expertise.

---

## 👶 Who Is Vulnerable?

Pneumonia can affect **anyone**, but the following groups are particularly vulnerable:

| Group | Risk Factors |
|-------|--------------|
| 👶 Infants and children under 5 | Underdeveloped immune systems |
| 👵 Older adults (65+) | Weakened immunity, chronic illnesses |
| 🧑‍⚕️ Immunocompromised patients | Cancer, HIV, organ transplants |
| 🚬 Smokers & alcoholics | Lung damage, lowered immunity |
| 🧭 People in low-income regions | Poor access to vaccines & antibiotics |

According to the **World Health Organization**, pneumonia accounts for:
- **>15% of all deaths** in children under 5 worldwide
- **>50,000 deaths per year** in the U.S. alone (CDC, 2015)

---

## 🎯 Project Objectives

- Automate the **detection and localization** of lung opacities associated with pneumonia
- Reduce the burden on radiologists by assisting in early diagnosis
- Create an **end-to-end machine learning pipeline** from data loading to prediction

---

## 📂 Dataset Summary

| Attribute | Description |
|----------|-------------|
| Source | RSNA, NIH, MD.ai, The Society of Thoracic Radiology |
| Format | DICOM images, bounding boxes in CSV |
| Train images | ~26,000 labeled chest X-rays |
| Annotation | Bounding boxes with labels (Target: 1 = pneumonia) |
| Evaluation | mAP (mean Average Precision) at IoU thresholds from 0.4 to 0.75 |

---

## 🧠 Model Architecture

| Component | Details |
|----------|---------|
| Preprocessing | DICOM to PNG/JPEG, resizing, normalization |
| Model | Faster R-CNN (PyTorch or TensorFlow implementation) |
| Input | Resized chest X-rays (1024×1024) |
| Output | Bounding boxes + confidence score |
| Training | Custom dataset loader, data augmentation |
| Inference | Batch prediction with non-max suppression (NMS) |

---

## ⚙️ Setup Guide

### 🔐 Step 1: Set up Kaggle API

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/account).
2. Click on "Create New API Token" – this downloads `kaggle.json`.
3. Place the file at:  



### 📦 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
numpy
pandas
pydicom
opencv-python
matplotlib
tqdm
torch
torchvision
scikit-learn

import os
os.environ['KAGGLE_CONFIG_DIR'] = 'C:/Users/YourUsername/.kaggle'

!kaggle competitions download -c rsna-pneumonia-detection-challenge
!unzip rsna-pneumonia-detection-challenge.zip -d data/

```

.
├── data/                           # Dataset files
├── src/
│   ├── preprocess.py               # DICOM preprocessing
│   ├── dataset.py                  # Custom PyTorch Dataset
│   ├── model.py                    # Faster R-CNN wrapper
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation code
├── notebooks/                      # EDA and experiments
├── outputs/                        # Saved models and logs
├── README.md
├── requirements.txt
└── .kaggle/kaggle.json             # Kaggle API key (not shared)


Refrences:
Rui P, Kang K. National Ambulatory Medical Care Survey: 2015 Emergency Department Summary Tables. CDC Link
Deaths: Final Data for 2015. National Vital Statistics Reports. CDC
Franquet T. Imaging of community-acquired pneumonia. J Thorac Imaging. 2018. PMID: 30036297
Kelly B. The Chest Radiograph. Ulster Med J 2012;81(3):143-148
Wang X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR 2017. PDF
Kaggle RSNA Challenge Overview. hello@reallygreatsite.com - www.reallygreatsite.com
