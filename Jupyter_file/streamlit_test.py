## script  to defing streamlit


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title('Pnemonia  Detection - multimodel Interface')


# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Select a model",
    ("FCNN", "Custom CNN", "MobileNet", "UNet")
)


