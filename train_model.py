import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data[
    [
        "attendance",
        "previous_marks",
        "assignment_marks",
        "study_hours",
        "internal_marks"
    ]
]

# Output
y = data["performance"]

# Convert text labels into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Model Training Completed!")
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "student_model.pkl")

# Save encoder
joblib.dump(encoder, "label_encoder.pkl")

print("Model saved successfully!")