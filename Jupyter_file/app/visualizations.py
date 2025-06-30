import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle

sns.set_style("whitegrid")
sns.set_palette("Set3")

# ----------------------
# 🔧 Utility Functions
# ----------------------
def sanitize_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    return df

def add_labels(ax):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.text(p.get_x() + p.get_width() / 2., h + 1, f"{int(h)}",
                    ha="center", va="bottom", fontsize=9)

def add_age_bins(df):
    if "PatientAge" in df.columns:
        bins = list(range(0, 101, 10))  # [0, 10, 20, ..., 100] → 11 values = 10 bins
        labels = [f"{i}–{i+9}" for i in range(0, 100, 10)]  # 10 labels
        df["PatientAgeBins"] = pd.cut(df["PatientAge"], bins=bins, labels=labels, include_lowest=True)


# ----------------------
# 📊 Plotting Functions
# ----------------------
def plot_categorical_distributions(df, columns, hue=None, width=15, show_distribution=True):
    total = len(df)
    fig, axes = plt.subplots(len(columns), 1, figsize=(width, len(columns) * 5), constrained_layout=True)
    if len(columns) == 1:
        axes = [axes]
    for i, col in enumerate(columns):
        ax = axes[i]
        sns.countplot(data=df, x=col, hue=hue, palette="Set2", ax=ax)
        ax.set_title(f"{col} Distribution")
        ax.tick_params(axis="x", rotation=45)
        if show_distribution:
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.text(p.get_x() + p.get_width() / 2., h + total * 0.005,
                            f'{100 * h / total:.1f}%', ha='center')
    st.pyplot(fig)

def plot_class_distribution(df):
    st.subheader("📊 Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="class", order=df["class"].value_counts().index, ax=ax)
    add_labels(ax)
    st.pyplot(fig)

def plot_target_class_combo(df):
    st.subheader("📊 Target vs Class")
    grouped = df.groupby("Target")["class"].value_counts().rename("count").reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=grouped, x="Target", y="count", hue="class", ax=ax)
    add_labels(ax)
    st.pyplot(fig)

def plot_target_distribution(df):
    st.subheader("🎯 Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Target", ax=ax)
    add_labels(ax)
    st.pyplot(fig)

def plot_gender_distribution(df):
    st.subheader("⚧️ Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="PatientSex", ax=ax)
    add_labels(ax)
    st.pyplot(fig)

def plot_top_ages(df):
    st.subheader("🎂 Top Patient Ages")
    top_ages = df["PatientAge"].value_counts().nlargest(25).index
    filtered = df[df["PatientAge"].isin(top_ages)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=filtered, x="PatientAge", order=top_ages, ax=ax)
    add_labels(ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_pie_by_target(df):
    st.subheader("🥧 Target Distribution")
    fig, ax = plt.subplots()
    df.drop_duplicates("patientId")["Target"].value_counts().plot.pie(
        labels=["Negative", "Positive"], autopct="%.0f%%", ax=ax)
    st.pyplot(fig)

def plot_pie_by_class(df):
    st.subheader("🥧 Class Distribution")
    fig, ax = plt.subplots()
    df["class"].value_counts().plot.pie(autopct="%.0f%%", ax=ax, ylabel="")
    st.pyplot(fig)

def plot_opacity_center_overlay(df):
    st.subheader("📌 Bounding Box Centers")
    sample = df[df["Target"] == 1].copy()
    sample = sample.sample(min(2000, len(sample)))
    sample["xc"] = sample["x"] + sample["width"] / 2
    sample["yc"] = sample["y"] + sample["height"] / 2
    fig, ax = plt.subplots()
    sample.plot.scatter(x="xc", y="yc", ax=ax, alpha=0.5, marker=".", color="red")
    for _, row in sample.iterrows():
        ax.add_patch(Rectangle((row["x"], row["y"]), row["width"], row["height"], alpha=0.003, color="yellow"))
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    st.pyplot(fig)

# ----------------------
# 🚀 Run Visualizations
# ----------------------
def run_visualizations():
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return

    df = sanitize_dataframe(st.session_state.df.copy())
    add_age_bins(df)

    st.title("📈 Pneumonia Visualizations")

    summary = {
        "Total Records": len(df),
        "Unique Patients": df["patientId"].nunique() if "patientId" in df.columns else "N/A",
        "Positive Cases": int(df["Target"].sum()) if "Target" in df.columns else "N/A",
        "Class Labels": ', '.join(df["class"].astype(str).unique()) if "class" in df.columns else "N/A"
    }
    st.markdown("### 📋 Dataset Summary")
    st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]))

    # Visualization options
    options = {
        "Class Distribution": plot_class_distribution,
        "Target vs Class": plot_target_class_combo,
        "Target Distribution": plot_target_distribution,
        "Gender Distribution": plot_gender_distribution,
        "Top Patient Ages": plot_top_ages,
        "Target Pie Chart": plot_pie_by_target,
        "Class Pie Chart": plot_pie_by_class,
        "Bounding Box Centers": plot_opacity_center_overlay,
        "Patient Age Group Distributions": None,
        "Custom Categorical Plot": None
    }

    choice = st.selectbox("Choose a visualization", list(options.keys()))

    if choice == "Patient Age Group Distributions":
        df_bins = df.drop_duplicates("patientId")
        st.subheader("Age Group Distribution (no hue)")
        plot_categorical_distributions(df_bins, ["PatientAgeBins"], hue=None, width=20)

        st.subheader("Age Group by Target")
        plot_categorical_distributions(df_bins, ["PatientAgeBins"], hue="Target", width=20)

        st.subheader("Age Group by Class")
        plot_categorical_distributions(df_bins, ["PatientAgeBins"], hue="class", width=20)

    elif choice == "Custom Categorical Plot":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        st.subheader("🧩 Build a Custom Plot")
        selected = st.multiselect("Select categorical columns", options=cat_cols, default=["PatientSex"])
        hue_selection = st.selectbox("Hue (optional)", ["None"] + df.columns.tolist())
        hue = None if hue_selection == "None" else hue_selection

        if selected:
            plot_categorical_distributions(
                df=df.drop_duplicates("patientId"),
                columns=selected,
                hue=hue,
                width=20
            )

    else:
        plot_func = options[choice]
        if plot_func:
            plot_func(df)
