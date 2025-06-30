import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Import page modules
from eda import run_eda
from visualizations import run_visualizations
from image_processing import run_image_viewer
from modeling import run_model_predictions  # ✅ New import

# --- Page Configuration ---
st.set_page_config(page_title="Pneumonia Detection Dashboard", layout="wide")

# --- Sidebar Navigation ---
def sidebar_navigation():
    return option_menu(
        menu_title="🔘 Navigation",
        options=[
            "🏠 Home",
            "📊 Executive Summary",
            "📁 Upload Data",
            "🔍 Exploratory Data Analysis",
            "📈 Visualizations",
            "🩻 Image Viewer",
            "🧠 Model Predictions"  # ✅ New menu item
        ],
        icons=["house", "bar-chart", "cloud-upload", "search", "graph-up", "image", "cpu"],
        menu_icon="cast",
        default_index=0
    )

with st.sidebar:
    selected = sidebar_navigation()

# --- File Upload ---
def file_upload_section():
    st.title("📁 Upload Dataset(s)")
    uploaded_files = st.file_uploader("Choose one or more CSV files", 
                                        type="csv", 
                                        accept_multiple_files=True, 
                                        key="csv_upload"
                                    )


    if uploaded_files:
        st.session_state.dfs = {
            f.name: pd.read_csv(f) for f in uploaded_files
        }
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

if selected == "📁 Upload Data":
    file_upload_section()

# --- Dataset Selection ---
def dataset_selection():
    if 'dfs' in st.session_state:
        dataset_names = list(st.session_state.dfs.keys())
        selected_dataset = st.sidebar.selectbox("📂 Select Dataset", dataset_names)
        df = st.session_state.dfs[selected_dataset]
        st.session_state.df = df
        st.session_state.selected_dataset = selected_dataset  # Store selected dataset globally
        return df
    return None

df = dataset_selection()

# --- Page Routing ---
def display_home_page():
    st.title("🏠 Welcome")
    st.write("This is a diagnostic dashboard built to support pneumonia detection through data insights and ML modeling.")

def display_executive_summary():
    st.title("📊 Executive Summary")
    st.caption("High-level metrics for stakeholders and decision-makers.")

    if df is not None:
        st.metric("🧑‍⚕️ Total Patients", df["patientId"].nunique() if "patientId" in df.columns else "N/A")
        if "Target" in df.columns:
            st.metric("🩺 Positive Cases", int(df["Target"].sum()))
            st.metric("✅ Negative Cases", int((df["Target"] == 0).sum()))
    else:
        st.warning("Please upload a dataset to see summary statistics.")

def display_eda():
    st.title("🔍 Exploratory Data Analysis")
    if df is not None and 'selected_dataset' in st.session_state:
        run_eda(df, st.session_state.selected_dataset)
    else:
        st.warning("Please upload and select a dataset first.")

def display_visualizations():
    st.title("📈 Visualizations")
    if df is not None:
        run_visualizations()
    else:
        st.warning("Please upload and select a dataset to explore visualizations.")

def display_image_viewer():
    st.title("🩻 Image + Mask Viewer")
    run_image_viewer()

def display_model_predictions():
    st.title("🧠 Model Predictions")
    run_model_predictions()

# --- Page Display Logic ---
if selected == "🏠 Home":
    display_home_page()

elif selected == "📊 Executive Summary":
    display_executive_summary()

    '''elif selected == "📁 Upload Data":
        file_upload_section()'''

elif selected == "🔍 Exploratory Data Analysis":
    display_eda()

elif selected == "📈 Visualizations":
    display_visualizations()

elif selected == "🩻 Image Viewer":
    display_image_viewer()

elif selected == "🧠 Model Predictions":
    display_model_predictions()
#dd