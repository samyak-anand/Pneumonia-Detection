# app/eda.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df, dataset_name):
    st.title("Exploratory Data Analysis")

    if df is None or df.empty:
        st.warning("Please upload a valid dataset.")
        return

    st.subheader(f"📄 Data Preview: {dataset_name}")
    st.dataframe(df.head())

    st.subheader("📊 Basic Dataset Information")
    st.write(f"Shape (rows, columns): {df.shape}")
    st.write("Data Types:")
    st.write(df.dtypes)

    st.subheader("❓ Missing Values")
    st.write("Count:")
    st.write(df.isnull().sum())
    st.write("Percentage:")
    st.write((df.isnull().mean() * 100).round(2))

    st.subheader("🔢 Unique Values per Column")
    st.write(df.nunique())
    if st.checkbox("Show unique values for each column"):
        for col in df.columns:
            st.write(f"{col} ({df[col].nunique()} unique values):")
            st.write(df[col].unique())

    st.subheader("🧬 Duplicate Records")
    st.write(f"Total Duplicates: {df.duplicated().sum()}")

    st.subheader("🧪 Inconsistent Data Types (Mixed types in columns)")
    for col in df.columns:
        unique_types = df[col].dropna().map(type).nunique()
        if unique_types > 1:
            st.warning(f"Column '{col}' has mixed data types!")

    st.subheader("📈 Numerical Feature Analysis")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if not num_cols.empty:
        st.write(df[num_cols].describe().T)

        selected_num_col = st.selectbox("Select numerical column for distribution plot", num_cols)
        if selected_num_col:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[selected_num_col], kde=True, ax=ax[0])
            ax[0].set_title(f"Histogram & KDE - {selected_num_col}")
            sns.boxplot(x=df[selected_num_col], ax=ax[1])
            ax[1].set_title(f"Boxplot - {selected_num_col}")
            st.pyplot(fig)

            # Outlier Detection
            q1 = df[selected_num_col].quantile(0.25)
            q3 = df[selected_num_col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[selected_num_col] < q1 - 1.5 * iqr) | (df[selected_num_col] > q3 + 1.5 * iqr)]
            st.write(f"Outliers detected in {selected_num_col}: {len(outliers)}")

    # Optional plot if relevant columns exist
    if "class" in df.columns and "sex" in df.columns:
        st.subheader("📊 Visual: Class Distribution by Gender")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="class", hue="sex", ax=ax)
        ax.set_title("Pneumonia Cases by Gender")
        st.pyplot(fig)

    st.subheader("🧑‍⚕️ Patient-Specific Insights")
    if "patientId" in df.columns:
        st.write(f"Total Unique Patients: {df['patientId'].nunique()}")

    if "age" in df.columns:
        st.write(f"Age: Min = {df['age'].min()}, Max = {df['age'].max()}, Mean = {df['age'].mean():.2f}")

    if "sex" in df.columns:
        st.write("Sex Distribution:")
        st.write(df["sex"].value_counts())

    if "Target" in df.columns:
        st.write("Target Class Distribution:")
        st.write(df["Target"].value_counts())

    if "class" in df.columns:
        st.write("Class Distribution:")
        st.write(df["class"].value_counts())

    if {"pixelSpacing_X", "pixelSpacing_Y"}.issubset(df.columns):
        st.write("Pixel Spacing Info:")
        st.write(f"X - Mean: {df['pixelSpacing_X'].mean():.2f}, Unique: {df['pixelSpacing_X'].nunique()}")
        st.write(f"Y - Mean: {df['pixelSpacing_Y'].mean():.2f}, Unique: {df['pixelSpacing_Y'].nunique()}")

    if {"x", "y", "width", "height"}.issubset(df.columns):
        st.write("Bounding Box Stats:")
        st.write(f"X: {df['x'].min()} to {df['x'].max()}")
        st.write(f"Y: {df['y'].min()} to {df['y'].max()}")
        st.write(f"Width: {df['width'].min()} to {df['width'].max()}")
        st.write(f"Height: {df['height'].min()} to {df['height'].max()}")

    if "viewPosition" in df.columns:
        st.write("View Position Distribution:")
        st.write(df["viewPosition"].value_counts())
