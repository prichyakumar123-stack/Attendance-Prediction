from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("student_model.pkl")
encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment_marks = float(request.form["assignment_marks"])
    study_hours = float(request.form["study_hours"])
    internal_marks = float(request.form["internal_marks"])

    # Create input data
    input_data = pd.DataFrame([
        [
            attendance,
            previous_marks,
            assignment_marks,
            study_hours,
            internal_marks
        ]
    ], columns=[
        "attendance",
        "previous_marks",
        "assignment_marks",
        "study_hours",
        "internal_marks"
    ])

    # Prediction
    prediction = model.predict(input_data)

    # Convert number back to label
    result = encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)