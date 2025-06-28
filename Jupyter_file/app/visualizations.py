import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

sns.set_palette("Set3")

def _add_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 1, f"{int(height)}",
                    ha="center", va="bottom", fontsize=9, rotation=90)
            
def plot_joint_hex_bounding_box_center(df):
    required = {"x", "y", "width", "height", "Target"}
    if required.issubset(df.columns):
        st.subheader("🔷 Bounding Box Center Heatmap (Target = 1)")
        bboxes = df[df["Target"] == 1].copy()
        bboxes["xw"] = bboxes["x"] + bboxes["width"] / 2
        bboxes["yh"] = bboxes["y"] + bboxes["height"] / 2

        with sns.axes_style("white"):
            g = sns.jointplot(
                x='xw', y='yh', data=bboxes,
                kind='hex', height=8, alpha=0.5
            )
        g.fig.suptitle("Bounding Box Location When There Is Evidence of Pneumonia")
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        st.pyplot(g.fig)

def plot_class_distribution(df):
    if "class" in df.columns:
        st.subheader("📊 Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='class', order=df['class'].value_counts().index, ax=ax)
        _add_labels(ax)
        st.pyplot(fig)

def plot_target_class_combo(df):
    if "Target" in df.columns and "class" in df.columns:
        st.subheader("📊 Target vs Class")
        tmp = df.groupby("Target")["class"].value_counts()
        df_combo = pd.DataFrame({"Exams": tmp.values}, index=tmp.index).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=df_combo, x="Target", y="Exams", hue="class", ax=ax)
        _add_labels(ax)
        st.pyplot(fig)

def plot_target_distribution(df):
    if "Target" in df.columns:
        st.subheader("🎯 Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Target", ax=ax)
        _add_labels(ax)
        st.pyplot(fig)

def plot_gender_distribution(df):
    if "PatientSex" in df.columns:
        st.subheader("⚧️ Gender Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="PatientSex", ax=ax)
        _add_labels(ax)
        st.pyplot(fig)

def plot_top_ages(df, top_n=25):
    if "PatientAge" in df.columns:
        st.subheader(f"🎂 Top {top_n} Most Common Patient Ages")
        top_ages = df["PatientAge"].value_counts().nlargest(top_n).index
        filtered = df[df["PatientAge"].isin(top_ages)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=filtered, x="PatientAge", order=top_ages, ax=ax)
        _add_labels(ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

def plot_pie_by_target(df):
    if "Target" in df.columns and "patientId" in df.columns:
        st.subheader("🥧 Pneumonia Class Distribution by Target")
        fig, ax = plt.subplots()
        df.drop_duplicates("patientId")["Target"].value_counts().plot.pie(
            labels=["Negative", "Positive"],
            autopct="%1.0f%%",
            ax=ax,
            figsize=(6, 6)
        )
        st.pyplot(fig)

def plot_pie_by_class(df):
    if "class" in df.columns:
        st.subheader("🥧 Pneumonia Class Distribution by Class")
        fig, ax = plt.subplots()
        df["class"].value_counts().sort_index(ascending=False).plot.pie(
            autopct="%.0f%%",
            labels=df["class"].value_counts().sort_index(ascending=False).index,
            ylabel="",
            ax=ax
        )
        st.pyplot(fig)

def plot_opacity_center_overlay(df):
    required = {"x", "y", "width", "height", "Target"}
    if required.issubset(df.columns):
        st.subheader("📌 Lung Opacity Bounding Box Centers")
        sample = df[df['Target'] == 1].copy()
        sample = sample.sample(min(2000, len(sample)))
        sample['xc'] = sample['x'] + sample['width'] / 2
        sample['yc'] = sample['y'] + sample['height'] / 2
        fig, ax = plt.subplots()
        sample.plot.scatter(x='xc', y='yc', ax=ax, marker='.', alpha=0.6, color='red')
        for _, row in sample.iterrows():
            ax.add_patch(Rectangle((row['x'], row['y']), row['width'], row['height'],
                                   alpha=0.0035, color="yellow"))
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        st.pyplot(fig)

def plot_target1_geometry_dist(df):
    required = {"x", "y", "width", "height", "Target"}
    if required.issubset(df.columns):
        st.subheader("📐 Bounding Box Distribution (Target=1)")
        t1 = df[df["Target"] == 1]
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        sns.histplot(t1["x"], kde=True, bins=50, color="red", ax=ax[0, 0])
        sns.histplot(t1["y"], kde=True, bins=50, color="blue", ax=ax[0, 1])
        sns.histplot(t1["width"], kde=True, bins=50, color="green", ax=ax[1, 0])
        sns.histplot(t1["height"], kde=True, bins=50, color="magenta", ax=ax[1, 1])
        ax[0, 0].set_title("X"), ax[0, 1].set_title("Y")
        ax[1, 0].set_title("Width"), ax[1, 1].set_title("Height")
        plt.tight_layout()
        st.pyplot(fig)

# --------------------
# Streamlit Entry Point
# --------------------
def run_visualizations():
    if 'df' not in st.session_state:
        st.warning("📂 Please upload and select a dataset.")
        return

    df = st.session_state.df

    st.title("📈 Visualizations")
    plot_class_distribution(df)
    plot_target_class_combo(df)
    plot_target_distribution(df)
    plot_gender_distribution(df)
    plot_top_ages(df)
    plot_pie_by_target(df)
    plot_pie_by_class(df)
    plot_opacity_center_overlay(df)
    plot_target1_geometry_dist(df)
    plot_joint_hex_bounding_box_center(df)

