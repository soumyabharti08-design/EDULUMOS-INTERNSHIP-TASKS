
import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Create the data folder
os.makedirs("data", exist_ok=True)

first_names = [
    "Aarav", "Priya", "Rahul", "Ananya", "Rohan",
    "Sneha", "Aditya", "Neha", "Arjun", "Kavya",
    "Ishaan", "Meera", "Karan", "Riya", "Dev",
    "Pooja", "Vikram", "Aisha", "Varun", "Nisha"
]

last_names = [
    "Sharma", "Verma", "Patel", "Singh", "Gupta",
    "Kumar", "Joshi", "Mishra", "Yadav", "Das"
]

number_of_students = 500

total_classes = np.random.randint(
    60, 121, size=number_of_students
)

attendance_rates = np.random.beta(
    5, 2, size=number_of_students
)

classes_attended = np.rint(
    total_classes * attendance_rates
).astype(int)

classes_attended = np.clip(
    classes_attended, 0, total_classes
)

attendance_percentage = (
    classes_attended / total_classes
) * 100

attendance_risk = np.where(
    attendance_percentage < 75,
    "At Risk",
    "Safe"
)

student_names = [
    f"{np.random.choice(first_names)} "
    f"{np.random.choice(last_names)}"
    for _ in range(number_of_students)
]

df = pd.DataFrame({
    "student_id": [
        f"STU{i:03d}"
        for i in range(1, number_of_students + 1)
    ],
    "student_name": student_names,
    "total_classes": total_classes,
    "classes_attended": classes_attended,
    "attendance_percentage": np.round(
        attendance_percentage, 2
    ),
    "attendance_risk": attendance_risk
})

df.to_csv("data/sample_attendance.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst five records:")
print(df.head())

print("\nAttendance risk counts:")
print(df["attendance_risk"].value_counts())

print("\nFile saved at: data/sample_attendance.csv")
