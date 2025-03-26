from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature names safely
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"❌ Error loading model, scaler, or feature names: {e}")
    model, scaler, feature_names = None, None, None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None
    ad_spend = None
    
    if request.method == "POST":
        try:
            if not model or not scaler or not feature_names:
                prediction = "⚠️ Model, Scaler, or feature names not loaded properly."
            else:
                # Get advertising spend value
                ad_spend = request.form.get("Advertising_Spend")
                if ad_spend is None:
                    error_message = "❌ Missing advertising spend value"
                else:
                    try:
                        ad_spend = float(ad_spend)
                        if ad_spend < 0:
                            error_message = "❌ Advertising spend cannot be negative"
                        else:
                            # Convert to DataFrame with feature names
                            input_df = pd.DataFrame([[ad_spend]], columns=["Advertising_Spend"])
                            
                            # Scale the input
                            input_scaled = scaler.transform(input_df)
                            
                            # Make prediction
                            pred_value = float(model.predict(input_scaled)[0])
                            
                            # Add confidence interval (95%)
                            std_dev = np.std(model.predict(input_scaled))
                            confidence_interval = round(1.96 * std_dev, 2)
                            
                            prediction = {
                                'value': round(pred_value, 2),
                                'ci': confidence_interval,
                                'ad_spend': ad_spend
                            }
                            
                    except ValueError:
                        error_message = "❌ Invalid input. Please enter a valid number."

        except Exception as e:
            error_message = f"❌ Error: {str(e)}"

    return render_template("index.html", 
                         prediction=prediction,
                         error_message=error_message,
                         ad_spend=ad_spend)

if __name__ == "__main__":
    app.run(debug=True)
