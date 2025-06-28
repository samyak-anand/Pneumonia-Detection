# -----------------------------
# Import necessary libraries
# -----------------------------
from streamlit_option_menu import option_menu  # For horizontal menu navigation
import streamlit as st  # Core Streamlit framework
import pandas as pd  # For data manipulation

# -----------------------------
# Configure Streamlit page layout
# -----------------------------
st.set_page_config(layout="wide")  # Set the layout to wide for more horizontal space

# -----------------------------
# Navigation bar setup using option_menu
# -----------------------------
# Provides horizontal navigation for multiple app sections
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Dashboard", "Visualizations", "Modeling"],
    icons=["house", "bar-chart", "graph-up", "cpu"],  # Icon names (not visible after removing emojis)
    orientation="horizontal"
)

# -----------------------------
# File uploader section
# -----------------------------
# Allow users to upload one or more CSV files
uploaded_files = st.file_uploader(
    label="Upload CSV Dataset(s)",
    type="csv",
    accept_multiple_files=True
)

# Store uploaded datasets in session_state for persistence
if uploaded_files and 'dfs' not in st.session_state:
    st.session_state.dfs = {
        uploaded_file.name: pd.read_csv(uploaded_file)
        for uploaded_file in uploaded_files
    }

# -----------------------------
# Sidebar dataset selector
# -----------------------------
# Allows users to choose one of the uploaded datasets
if 'dfs' in st.session_state:
    dataset_names = list(st.session_state.dfs.keys())
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_names)
    df = st.session_state.dfs[selected_dataset]
else:
    df = None  # If no dataset is uploaded yet

# -----------------------------
# Home Page
# -----------------------------
if selected == "Home":
    st.title("Home")
    st.write("Welcome to the Pneumonia Detection Dashboard. Use the tabs above to explore, visualize, and analyze your data.")

# -----------------------------
# EDA (Exploratory Data Analysis) Page
# -----------------------------
elif selected == "EDA":
    st.title("Exploratory Data Analysis")

    # Check if a dataset is selected and available
    if df is not None:
        # Preview the first few rows of the dataset
        st.subheader(f"Data Preview: {selected_dataset}")
        st.dataframe(df.head())

        # Show basic dataset info including shape and column data types
        st.subheader("Basic Dataset Information")
        st.write(f"Shape (rows, columns): {df.shape}")
        st.write("Data Types:")
        st.write(df.dtypes)

        # Display missing values per column
        st.subheader("Missing Values Count")
        st.write(df.isnull().sum())

        # Show percentage of missing values
        st.subheader("Missing Values Percentage")
        st.write((df.isnull().mean() * 100).round(2))

        # Display count of unique values per column
        st.subheader("Unique Values per Column")
        st.write(df.nunique())

        # Optional toggle to display actual unique values in each column
        if st.checkbox("Show unique values for each column"):
            for col in df.columns:
                st.write(f"{col} ({df[col].nunique()} unique values):")
                st.write(df[col].unique())

        # ----------------------------------------
        # Summary statistics related to patients
        # ----------------------------------------
        st.subheader("Patient Information Summary")

        # Total number of unique patients
        if "patientId" in df.columns:
            st.write(f"Total Unique Patients: {df['patientId'].nunique()}")

        # Age statistics: min, max, mean
        if "age" in df.columns:
            st.write("Age Statistics:")
            st.write(f"Minimum Age: {df['age'].min()}")
            st.write(f"Maximum Age: {df['age'].max()}")
            st.write(f"Average Age: {df['age'].mean():.2f}")

        # Sex distribution
        if "sex" in df.columns:
            st.write("Sex Distribution:")
            st.write(df["sex"].value_counts())

        # Target class distribution (e.g., pneumonia detection)
        if "Target" in df.columns:
            st.write("Target Class Distribution:")
            st.write(df["Target"].value_counts())

        # Additional class information if present
        if "class" in df.columns:
            st.write("Class Distribution:")
            st.write(df["class"].value_counts())

        # Pixel spacing statistics
        if "pixelSpacing_X" in df.columns and "pixelSpacing_Y" in df.columns:
            st.write("Pixel Spacing Information:")
            st.write(f"X - Mean: {df['pixelSpacing_X'].mean():.2f}, Unique: {df['pixelSpacing_X'].nunique()}")
            st.write(f"Y - Mean: {df['pixelSpacing_Y'].mean():.2f}, Unique: {df['pixelSpacing_Y'].nunique()}")

        # Bounding box statistics (for object detection)
        if all(col in df.columns for col in ["x", "y", "width", "height"]):
            st.write("Bounding Box Statistics:")
            st.write(f"X Range: {df['x'].min()} to {df['x'].max()}")
            st.write(f"Y Range: {df['y'].min()} to {df['y'].max()}")
            st.write(f"Width Range: {df['width'].min()} to {df['width'].max()}")
            st.write(f"Height Range: {df['height'].min()} to {df['height'].max()}")

        # View position analysis (AP vs PA)
        if "viewPosition" in df.columns:
            st.write("View Position Distribution:")
            st.write(df["viewPosition"].value_counts())

            ap_count = df[df["viewPosition"] == "AP"].shape[0]
            pa_count = df[df["viewPosition"] == "PA"].shape[0]
            st.write("AP vs PA View Counts:")
            st.write(f"AP (Anteroposterior): {ap_count}")
            st.write(f"PA (Posteroanterior): {pa_count}")

    else:
        # If no dataset is loaded
        st.warning("Please upload at least one dataset on the Home page.")

# -----------------------------
# Visualization Page
# -----------------------------
elif selected == "Visualizations":
    st.title("Visualizations")
    import matplotlib.pyplot as plt  # Imported only when needed

    # Check if dataset is stored in session state
    if 'df' in st.session_state:
        df = st.session_state.df

        # Ensure necessary columns are present
        if "Target" in df.columns and "patientId" in df.columns:
            st.subheader("Pneumonia Class Distribution (Pie Chart)")

            # Avoid duplicate patient entries
            class_counts = df.drop_duplicates("patientId")["Target"].value_counts()

            # Create pie chart for Target distribution
            fig, ax = plt.subplots()
            class_counts.plot.pie(
                labels=["Negative", "Positive"],
                autopct="%1.0f%%",
                figsize=(6, 6),
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.error("The dataset must contain 'Target' and 'patientId' columns.")
    else:
        st.warning("Please upload a dataset on the Home page.")

# -----------------------------
# Modeling Page Placeholder
# -----------------------------
elif selected == "Modeling":
    st.title("Modeling")
    st.info("Modeling tools such as training, prediction, and evaluation will be added in future versions.")
