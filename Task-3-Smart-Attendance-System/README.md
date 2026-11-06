
# Smart Attendance System Using Machine Learning

## Project Overview

The Smart Attendance System is a Python-based application that analyzes student attendance records and identifies students whose current attendance is below a specified threshold.

It uses data analysis, visualizations, a Decision Tree classification model, and an interactive Streamlit dashboard.

**Note:** The included dataset is synthetic and intended for demonstration and testing. The model demonstrates classification of current attendance patterns, not prediction of future attendance.

## Features

- Generate and analyze sample student attendance records.
- Validate and clean attendance data.
- Calculate attendance percentages from class counts.
- Identify students below the attendance threshold.
- Visualize attendance distributions and risk categories.
- Train and evaluate a Decision Tree classifier.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Upload attendance data through the Streamlit dashboard.
- Search student records and filter students currently at risk.
- Download attendance results as a CSV file.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
Smart-Attendance-System/
├── smart_attendance_dashboard.py
├── train_model.py
├── generate_data.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_attendance.csv
└── models/
    └── attendance_model.joblib
```


## Installation

1. Install Python 3.12 or another compatible Python version.
2. Clone or download this repository.
3. Open the project folder in VS Code.
4. Open the terminal in the project folder.
5. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Generate Sample Data

Run:

```bash
python generate_data.py
```

This creates `data/sample_attendance.csv` with 500 synthetic student records.

## Train and Evaluate the Model

Run:

```bash
python train_model.py
```

The script cleans the data, trains the classifier, prints evaluation metrics, generates charts, and saves the model in the `models` folder.

## Run the Dashboard

Run:

```bash
streamlit run smart_attendance_dashboard.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

The dashboard supports the sample dataset and CSV uploads containing the required columns:

- `student_id`
- `total_classes`
- `classes_attended`

An optional `student_name` column can be included.

## Attendance Calculation

Attendance percentage is calculated as:

Attendance Percentage = (Classes Attended / Total Classes) × 100

The illustrative threshold is 75%. Students below it are classified as currently at risk. Adjust the threshold to match your institution's policy.

## Model Evaluation

The training script reports accuracy, precision, recall, F1-score, and a confusion matrix.

The model uses `total_classes` and `classes_attended` to classify current attendance status. Since the target label is derived from attendance counts, this is a demonstration of classification rather than a genuine forecast of future attendance.

## Limitations

- The sample dataset is synthetic and does not represent real students.
- Evaluation metrics on synthetic data do not establish real-world performance.
- The model does not predict future attendance.
- Real-world forecasting requires suitable historical data and a clearly defined future outcome.
- Attendance risk should be reviewed alongside institutional policies and relevant context.

## Future Improvements

- Use an approved real-world attendance dataset.
- Predict future attendance using historical time-based features.
- Compare multiple classification algorithms.
- Add date-wise attendance tracking.
- Improve model validation and monitoring.

## Author

Developed as an educational Machine Learning project for internship learning and demonstration.
