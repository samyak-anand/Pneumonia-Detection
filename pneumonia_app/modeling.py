import streamlit as st
import numpy as np
from PIL import Image
import pydicom
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Default preprocessor
def default_preprocess(x):
    return x / 255.0

# Preprocessing function map
MODEL_PREPROCESSORS = {
    "FCNN": default_preprocess,
    "Custom CNN": default_preprocess,
    "MobileNet": mobilenet_preprocess,
    "U-Net": default_preprocess,
    "ResNet U-Net": resnet_preprocess,
    "MobileNet U-Net": mobilenet_preprocess
}

# Model loader
@st.cache_resource
def load_cnn_model(model_name):
    model_paths = {
        "FCNN": "models/fcnn_model.keras",
        "Custom CNN": "models/custom_cnn.keras",
        "MobileNet": "models/mobilenet_model.keras",
        "U-Net": "models/unet_model.keras",
        "ResNet U-Net": r"C:\Users\samya\PyCharmProject\Pneumonia-Detection\Jupyter_file\app\models\resnet50_unet_best.keras",
        "MobileNet U-Net": r"C:\Users\samya\PyCharmProject\Pneumonia-Detection\Jupyter_file\app\models\mobilenet_unet_best.keras"
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
            # Load and display image
            if uploaded_file.name.lower().endswith(".dcm"):
                ds = pydicom.dcmread(uploaded_file)
                image_array = ds.pixel_array
                image = Image.fromarray(image_array).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")

            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load model and determine input shape
            model = load_cnn_model(model_name)
            model_input_shape = model.input_shape[1:3]

            # Resize image
            image_resized = image.resize(model_input_shape)
            img_array = np.array(image_resized).astype(np.float32)

            # Preprocess based on model
            preprocessor = MODEL_PREPROCESSORS[model_name]
            img_array = preprocessor(img_array)  # Safe now that it's float32
            img_batch = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_batch)
            confidence = float(prediction[0][0])
            label = "Pneumonia" if confidence > 0.5 else "Normal"

            st.success(f"Prediction: **{label}**")
            st.write(f"Confidence Score: `{confidence:.4f}`")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
