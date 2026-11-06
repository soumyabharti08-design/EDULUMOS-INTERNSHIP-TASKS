
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📊",
    layout="wide"
)

THRESHOLD = 75

st.title("Smart Attendance System")
st.caption(
    "Student attendance analysis and Machine Learning demonstration"
)

st.info(
    "This project uses synthetic demonstration data. "
    "Model predictions classify current attendance patterns; "
    "they do not predict future attendance."
)

# -------------------------------
# Load the saved model
# -------------------------------

MODEL_PATH = os.path.join(
    "models", "attendance_model.joblib"
)

model_bundle = None

if os.path.exists(MODEL_PATH):
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(f"Could not load the trained model: {error}")
else:
    st.warning(
        "Trained model not found. Run "
        "`python train_model.py` from the project folder first."
    )

# -------------------------------
# Upload or use sample dataset
# -------------------------------

st.sidebar.header("Attendance Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload attendance CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV uploaded successfully.")
    except Exception as error:
        st.error(f"Unable to read this CSV file: {error}")
        st.stop()
else:
    sample_path = os.path.join(
        "data", "sample_attendance.csv"
    )

    if not os.path.exists(sample_path):
        st.error(
            "Sample dataset not found. "
            "Check data/sample_attendance.csv."
        )
        st.stop()

    df = pd.read_csv(sample_path)
    st.sidebar.caption("Showing the sample dataset.")

# -------------------------------
# Validate dataset columns
# -------------------------------

required_columns = [
    "student_id",
    "total_classes",
    "classes_attended"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Your CSV is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# -------------------------------
# Clean and validate records
# -------------------------------

original_count = len(df)

df["student_id"] = df["student_id"].astype("string").str.strip()

df["total_classes"] = pd.to_numeric(
    df["total_classes"], errors="coerce"
)

df["classes_attended"] = pd.to_numeric(
    df["classes_attended"], errors="coerce"
)

df = df.dropna(
    subset=[
        "student_id",
        "total_classes",
        "classes_attended"
    ]
)

df = df[
    (df["student_id"] != "")
    & (df["total_classes"] > 0)
    & (df["classes_attended"] >= 0)
    & (df["classes_attended"] <= df["total_classes"])
].copy()

df = df.drop_duplicates(
    subset=["student_id"]
).copy()

if df.empty:
    st.error(
        "No valid records found. Check class counts "
        "and student IDs in your CSV."
    )
    st.stop()

if len(df) < original_count:
    st.warning(
        f"{original_count - len(df)} invalid or duplicate "
        "records were removed."
    )

df["total_classes"] = df["total_classes"].astype(int)
df["classes_attended"] = df["classes_attended"].astype(int)

# Always calculate percentages from actual class counts.
df["attendance_percentage"] = (
    df["classes_attended"] / df["total_classes"]
) * 100

# Actual current attendance status.
df["actual_status"] = (
    df["attendance_percentage"] < THRESHOLD
).map({True: "At Risk", False: "Safe"})

# -------------------------------
# Summary metrics
# -------------------------------

st.subheader("Attendance Overview")

average_attendance = df["attendance_percentage"].mean()

at_risk_count = (
    df["actual_status"] == "At Risk"
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", len(df))

col2.metric(
    "Average Attendance",
    f"{average_attendance:.2f}%"
)

col3.metric("Students At Risk", int(at_risk_count))

col4.metric(
    "Students Safe",
    int(len(df) - at_risk_count)
)

st.caption(
    f"Current status uses an illustrative threshold of "
    f"{THRESHOLD}%. Change this threshold in the code "
    "if your institution uses a different rule."
)

# -------------------------------
# Attendance charts
# -------------------------------

st.subheader("Attendance Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="attendance_percentage",
        bins=20,
        kde=True,
        ax=ax
    )

    ax.axvline(
        THRESHOLD,
        linestyle="--",
        label=f"{THRESHOLD}% threshold"
    )

    ax.set_title("Attendance Percentage Distribution")
    ax.set_xlabel("Attendance Percentage")
    ax.set_ylabel("Number of Students")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

with chart2:
    status_counts = df["actual_status"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=status_counts.index,
        y=status_counts.values,
        ax=ax
    )

    ax.set_title("Current Attendance Risk")
    ax.set_xlabel("Actual Attendance Status")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)
    plt.close(fig)

# -------------------------------
# Machine Learning predictions
# -------------------------------

st.subheader("Machine Learning Classification")

st.write(
    "The saved model classifies current attendance records "
    "as Safe or At Risk. These predictions are separate "
    "from the actual status calculated using the threshold."
)

if model_bundle is not None:
    try:
        model = model_bundle["model"]

        features = model_bundle.get(
            "features",
            ["total_classes", "classes_attended"]
        )

        if all(column in df.columns for column in features):
            df["predicted_label"] = model.predict(
                df[features]
            )

            df["model_prediction"] = (
                df["predicted_label"]
                .map({0: "Safe", 1: "At Risk"})
            )

            st.caption(
                "Predictions are generated from the saved model. "
                "Because this model was trained on synthetic data, "
                "do not use its predictions for real student decisions."
            )

        else:
            st.warning(
                "The uploaded dataset does not contain all "
                "features required by the saved model."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

else:
    st.warning(
        "Model predictions are unavailable until a trained "
        "model has been saved."
    )

# -------------------------------
# Student records
# -------------------------------

st.subheader("Student Attendance Records")

search_text = st.text_input(
    "Search by Student ID or Name"
)

show_at_risk = st.checkbox(
    "Show only students currently at risk"
)

filtered_df = df.copy()

if search_text:
    search_mask = filtered_df["student_id"].astype(str).str.contains(
        search_text, case=False, na=False
    )

    if "student_name" in filtered_df.columns:
        search_mask |= filtered_df["student_name"].astype(
            str
        ).str.contains(search_text, case=False, na=False)

    filtered_df = filtered_df[search_mask]

if show_at_risk:
    filtered_df = filtered_df[
        filtered_df["actual_status"] == "At Risk"
    ]

display_columns = [
    column for column in [
        "student_id",
        "student_name",
        "total_classes",
        "classes_attended",
        "attendance_percentage",
        "actual_status",
        "model_prediction"
    ]
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns].round(2),
    use_container_width=True,
    hide_index=True
)

# -------------------------------
# Download cleaned records
# -------------------------------

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Attendance Results as CSV",
    data=csv_data,
    file_name="attendance_results.csv",
    mime="text/csv"
)

st.caption(
    "Smart Attendance System | Built with Python, "
    "Pandas, Scikit-learn, Seaborn and Streamlit"
)
