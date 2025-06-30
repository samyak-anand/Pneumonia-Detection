import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pydicom

st.title("🩺 Pneumonia Detection - Multi-Model Inference")

model_choice = st.sidebar.selectbox(
    "Select a model",
    ("FCNN", "Custom CNN", "MobileNet", "UNet")
)
from tensorflow.keras.applications.mobilenet import preprocess_input

model = tf.keras.models.load_model("mobilenet_model.h5", custom_objects={'preprocess_input': preprocess_input})

uploaded_file = st.file_uploader("Upload a chest X-ray image (.jpg, .png, .dcm)", type=["jpg", "jpeg", "png", "dcm"])

@st.cache_resource
def load_model(model_name):
    model_paths = {
        "FCNN": "fcnn_model.h5",
        "Custom CNN": "custom_cnn_model.h5",
        "MobileNet": "mobilenet_model.h5",
        "UNet": "mn_cnn_model.h5"
    }
    return tf.keras.models.load_model(model_paths[model_name], compile=False)

def preprocess_image(image, model_name):
    if model_name in ["FCNN", "Custom CNN", "MobileNet"]:
        image = image.resize((64, 64)).convert("RGB")
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    elif model_name == "UNet":
        image = image.resize((64, 64)).convert("RGB")
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)


if uploaded_file is not None:
    if uploaded_file.name.endswith(".dcm"):
        ds = pydicom.dcmread(uploaded_file)
        pixel_array = ds.pixel_array
        image = Image.fromarray((pixel_array / np.max(pixel_array) * 255).astype(np.uint8)).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image, model_choice)
    model = load_model(model_choice)
    prediction = model.predict(processed_image)

    st.write("### Prediction Result:")
    if model_choice == "UNet":
        st.image(prediction[0, :, :, 0], caption="Segmentation Mask", use_column_width=True)
    else:
        confidence = prediction[0][0]
        if confidence > 0.5:
            st.error(f"⚠️ Pneumonia Detected (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ No Pneumonia Detected (Confidence: {1 - confidence:.2f})")
