import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv("CareerAdvisory.csv")
df.columns = df.columns.str.strip()

# Fill missing expected columns if not present
df['Soft_Skills'] = df.get('Soft_Skills', 7)
df['Thinking_Ability'] = df.get('Thinking_Ability', 6)

# Encode categorical columns
label_encoders = {}
categorical_cols = ['Stream_in_12th', 'Entrance_Exam', 'Subject_Strength',
                    'Scholarship_Eligibility', 'Study_Abroad_Plan', 'Target_College',
                    'Career_Interest', 'Backup_Course', 'Counselor_Recommendation',
                    'Interest_Domain', 'Target_Job_Role']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and targets
features = ['Stream_in_12th', 'Entrance_Exam', 'Subject_Strength', '12th_Percentage',
            'Aptitude_Test_Score', 'Scholarship_Eligibility', 'Study_Abroad_Plan',
            'Interest_Domain', 'Soft_Skills', 'Thinking_Ability']

X = df[features]
y_college = df["Target_College"]
y_career = df["Career_Interest"]
y_backup = df["Backup_Course"]
y_counsel = df["Counselor_Recommendation"]
y_jobrole = df["Target_Job_Role"]

# Train models
model_college = DecisionTreeClassifier().fit(X, y_college)
model_career = DecisionTreeClassifier().fit(X, y_career)
model_backup = DecisionTreeClassifier().fit(X, y_backup)
model_counsel = DecisionTreeClassifier().fit(X, y_counsel)
model_jobrole = DecisionTreeClassifier().fit(X, y_jobrole)

# Save models and encoders
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model_college, "saved_models/model_college.pkl")
joblib.dump(model_career, "saved_models/model_career.pkl")
joblib.dump(model_backup, "saved_models/model_backup.pkl")
joblib.dump(model_counsel, "saved_models/model_counsel.pkl")
joblib.dump(model_jobrole, "saved_models/model_jobrole.pkl")
joblib.dump(label_encoders, "saved_models/label_encoders.pkl")

print("✅ Models and label encoders have been saved to 'saved_models/' folder.")
