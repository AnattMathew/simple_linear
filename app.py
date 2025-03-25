from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # ✅ Import Pandas

app = Flask(__name__)

# Load model and scaler safely
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None  # Set to None if loading fails

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            if not model or not scaler:
                prediction = "Model or Scaler not loaded properly."
            else:
                ad_spend = float(request.form["advertising_spend"])

                # ✅ Fix: Convert to DataFrame with feature names
                ad_spend_df = pd.DataFrame([[ad_spend]], columns=["advertising_spend"])
                ad_spend_scaled = scaler.transform(ad_spend_df)

                prediction = round(model.predict(ad_spend_scaled)[0], 2)
        except ValueError:
            prediction = "Invalid input. Please enter a valid number."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
