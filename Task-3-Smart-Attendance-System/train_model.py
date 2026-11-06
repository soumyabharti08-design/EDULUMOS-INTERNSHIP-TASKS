
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# 1. File paths and configuration
# -------------------------------

DATA_PATH = "data/sample_attendance.csv"
MODEL_DIR = "models"
THRESHOLD = 75

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 2. Load the dataset
# -------------------------------

print("\nLoading attendance dataset...")

df = pd.read_csv(DATA_PATH)

print("Original number of records:", len(df))
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# -------------------------------
# 3. Clean and validate data
# -------------------------------

required_columns = [
    "student_id",
    "total_classes",
    "classes_attended"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column is missing: {column}"
        )

# Remove duplicate student IDs.
df = df.drop_duplicates(subset=["student_id"])

# Convert class counts to numeric values.
df["total_classes"] = pd.to_numeric(
    df["total_classes"], errors="coerce"
)

df["classes_attended"] = pd.to_numeric(
    df["classes_attended"], errors="coerce"
)

# Remove records with missing or invalid counts.
df = df.dropna(
    subset=["student_id", "total_classes", "classes_attended"]
)

df = df[
    (df["student_id"].astype(str).str.strip() != "")
    & (df["total_classes"] > 0)
    & (df["classes_attended"] >= 0)
    & (df["classes_attended"] <= df["total_classes"])
].copy()

if df.empty:
    raise ValueError(
        "No valid attendance records remain after cleaning."
    )

df["total_classes"] = df["total_classes"].astype(int)
df["classes_attended"] = df["classes_attended"].astype(int)

# Calculate the percentage from the class counts.
# Do not trust a previously supplied percentage.
df["attendance_percentage"] = (
    df["classes_attended"] / df["total_classes"]
) * 100

# Determine actual current attendance status.
df["attendance_risk"] = (
    df["attendance_percentage"] < THRESHOLD
).map({True: "At Risk", False: "Safe"})

df = df.reset_index(drop=True)

print("\nRecords after cleaning:", len(df))
print("\nCurrent attendance categories:")
print(df["attendance_risk"].value_counts())

# -------------------------------
# 4. Exploratory data analysis
# -------------------------------

print(
    "\nAverage attendance:",
    round(df["attendance_percentage"].mean(), 2),
    "%"
)

print(
    "Students currently at risk:",
    (df["attendance_risk"] == "At Risk").sum()
)

# Attendance distribution chart.
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="attendance_percentage",
    bins=20,
    kde=True
)

plt.axvline(
    THRESHOLD,
    linestyle="--",
    label="75% threshold"
)

plt.title("Student Attendance Distribution")
plt.xlabel("Attendance Percentage")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()

plt.savefig("attendance_distribution.png")
plt.close()

# Risk category chart.
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="attendance_risk"
)

plt.title("Current Attendance Risk Categories")
plt.xlabel("Attendance Status")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig("attendance_risk.png")
plt.close()

print("\nCharts saved:")
print("- attendance_distribution.png")
print("- attendance_risk.png")

# -------------------------------
# 5. Prepare Machine Learning data
# -------------------------------

# Target:
# 1 = At Risk, 0 = Safe
#
# The target is derived from current attendance.
# Therefore, this is a demonstration classifier,
# NOT a model of future attendance.

X = df[["total_classes", "classes_attended"]].copy()

y = (
    df["attendance_risk"] == "At Risk"
).astype(int)

if y.nunique() < 2:
    raise ValueError(
        "The dataset must contain both Safe and At Risk students."
    )

# Split the records before fitting the model.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------
# 6. Train the classifier
# -------------------------------

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -------------------------------
# 7. Evaluate the model
# -------------------------------

y_pred = model.predict(X_test)

print("\nMODEL EVALUATION")
print("------------------------------")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print(
    "Precision:",
    round(
        precision_score(y_test, y_pred, zero_division=0),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(y_test, y_pred, zero_division=0),
        4
    )
)

print(
    "F1-score:",
    round(
        f1_score(y_test, y_pred, zero_division=0),
        4
    )
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Safe", "At Risk"],
        zero_division=0
    )
)

# Confusion matrix.
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "At Risk"],
    yticklabels=["Safe", "At Risk"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved: confusion_matrix.png")

# -------------------------------
# 8. Save model and metadata
# -------------------------------

model_bundle = {
    "model": model,
    "features": ["total_classes", "classes_attended"],
    "threshold": THRESHOLD,
    "target_mapping": {
        0: "Safe",
        1: "At Risk"
    },
    "training_data_type": "synthetic demonstration data",
    "prediction_scope": "current attendance classification"
}

joblib.dump(
    model_bundle,
    os.path.join(MODEL_DIR, "attendance_model.joblib")
)

print("\nModel saved successfully!")
print("Location: models/attendance_model.joblib")

print(
    "\nIMPORTANT: The dataset is synthetic. "
    "Evaluation metrics demonstrate the workflow only "
    "and do not establish real-world predictive performance."
)
