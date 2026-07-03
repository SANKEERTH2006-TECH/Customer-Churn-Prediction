from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize Flask App
app = Flask(__name__)

# -------------------------------
# Load Trained Model
# -------------------------------
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

# -------------------------------
# Load Saved Encoders
# -------------------------------
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Collect Input Data
    input_data = {
        "gender": request.form["gender"],
        "SeniorCitizen": int(request.form["SeniorCitizen"]),
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "tenure": int(request.form["tenure"]),
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }

    # Convert Dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode Categorical Features
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    # Arrange Columns in Correct Order
    input_df = input_df[feature_names]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    churn_probability = round(probability[0][1] * 100, 2)
    no_churn_probability = round(probability[0][0] * 100, 2)

    if prediction == 1:
        result = "Customer is likely to Churn"
    else:
        result = "Customer is likely to Stay"

    return render_template(
        "index.html",
        prediction_text=result,
        churn_prob=churn_probability,
        no_churn_prob=no_churn_probability
    )


# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)