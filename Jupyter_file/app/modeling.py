import streamlit as st
import numpy as np
from PIL import Image
import pydicom  # Make sure this is installed: pip install pydicom
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

@st.cache_resource
def load_cnn_model(model_name):
    model_paths = {
        "FCNN": "models/fcnn_model.keras",
        "Custom CNN": "models/custom_cnn.keras",
        "MobileNet": "models/mobilenet_model.keras",
        "U-Net": "models/unet_model.keras",
        "ResNet U-Net": "models/resnet_unet_model.keras",
        "MobileNet U-Net": "models/mobilenet_unet_model.keras"
    }

    if model_name not in model_paths:
        raise ValueError(f"Model '{model_name}' not found.")
    
    return load_model(model_paths[model_name])

def run_model_predictions():
    st.title("🧠 CNN Model Predictions")
    st.markdown("Upload a chest X-ray image and select a model to predict pneumonia presence.")

    model_name = st.selectbox("Select a Model", [
        "FCNN", "Custom CNN", "MobileNet", "U-Net", "ResNet U-Net", "MobileNet U-Net"
    ])

    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png", "dcm"], key="xray_upload")

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".dcm"):
                ds = pydicom.dcmread(uploaded_file)
                image_array = ds.pixel_array
                image = Image.fromarray(image_array).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")

            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess image
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = preprocess_input(img_array)
            img_batch = np.expand_dims(img_array, axis=0)

            # Load model
            model = load_cnn_model(model_name)

            # Predict
            prediction = model.predict(img_batch)
            confidence = float(prediction[0][0])
            label = "Pneumonia" if confidence > 0.5 else "Normal"

            st.success(f"Prediction: **{label}**")
            st.write(f"Confidence Score: `{confidence:.4f}`")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
