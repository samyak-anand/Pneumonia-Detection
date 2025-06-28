import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Page modules
from eda import run_eda
from visualizations import run_visualizations
from modeling import run_modeling

# --- Page Config ---
st.set_page_config(page_title="Pneumonia Detection Dashboard", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="🔘 Navigation",
        options=[
            "🏠 Home",
            "📊 Executive Summary",
            "📁 Upload Data",
            "🔍 Exploratory Data Analysis",
            "📈 Visualizations",
            "🤖 Modeling"
        ],
        icons=["house", "bar-chart", "cloud-upload", "search", "graph-up", "cpu"],
        menu_icon="cast",
        default_index=0
    )

# --- File Upload ---
if selected == "📁 Upload Data":
    st.title("📁 Upload Dataset(s)")
    uploaded_files = st.file_uploader("Choose one or more CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        st.session_state.dfs = {
            f.name: pd.read_csv(f) for f in uploaded_files
        }
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

# --- Dataset Selection ---
if 'dfs' in st.session_state:
    dataset_names = list(st.session_state.dfs.keys())
    selected_dataset = st.sidebar.selectbox("📂 Select Dataset", dataset_names)
    df = st.session_state.dfs[selected_dataset]
    st.session_state.df = df
else:
    df = None

# --- Page Routing ---
if selected == "🏠 Home":
    st.title("🏠 Welcome")
    st.write("This is a diagnostic dashboard built to support pneumonia detection through data insights and ML modeling.")

elif selected == "📊 Executive Summary":
    st.title("📊 Executive Summary")
    st.caption("High-level metrics for stakeholders and decision-makers.")

    if df is not None:
        st.metric("🧑‍⚕️ Total Patients", df["patientId"].nunique() if "patientId" in df.columns else "N/A")
        if "Target" in df.columns:
            st.metric("🩺 Positive Cases", int(df["Target"].sum()))
            st.metric("✅ Negative Cases", int((df["Target"] == 0).sum()))
    else:
        st.warning("Please upload a dataset to see summary statistics.")

elif selected == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    if df is not None:
        run_eda(df, selected_dataset)
    else:
        st.warning("Please upload and select a dataset first.")

elif selected == "📈 Visualizations":
    st.title("📈 Visualizations")
    if df is not None:
        run_visualizations()
    else:
        st.warning("Please upload and select a dataset to explore visualizations.")

elif selected == "🤖 Modeling":
    st.title("🤖 Pneumonia Prediction Modeling")
    if df is not None:
        run_modeling()
    else:
        st.warning("Please upload and select a dataset to proceed with modeling.")
