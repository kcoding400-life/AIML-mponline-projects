import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "car_price_model.pkl")
META_PATH = os.path.join("model", "metadata.json")

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

CURRENT_YEAR = metadata["year_max"]


def build_features(form):
    year = int(form["year"])
    car_age = CURRENT_YEAR - year
    return pd.DataFrame([{
        "Present_Price": float(form["present_price"]),
        "Kms_Driven": float(form["kms_driven"]),
        "Car_Age": car_age,
        "Owner": int(form["owner"]),
        "Brand": form["brand"],
        "Fuel_Type": form["fuel_type"],
        "Seller_Type": form["seller_type"],
        "Transmission": form["transmission"],
    }])


@app.route("/")
def home():
    return render_template("index.html", meta=metadata, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_features(request.form)
        pred_lakhs = float(model.predict(X)[0])
        pred_lakhs = max(0.0, round(pred_lakhs, 2))
        result = {
            "selling_price_lakhs": pred_lakhs,
            "selling_price_inr": int(pred_lakhs * 100000),
        }
        return render_template("index.html", meta=metadata, prediction=result,
                                form=request.form)
    except Exception as e:
        return render_template("index.html", meta=metadata, prediction=None,
                                error=str(e), form=request.form)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST the same fields as the form and get a JSON prediction back."""
    try:
        data = request.get_json(force=True)
        X = build_features(data)
        pred_lakhs = float(model.predict(X)[0])
        pred_lakhs = max(0.0, round(pred_lakhs, 2))
        return jsonify({
            "selling_price_lakhs": pred_lakhs,
            "selling_price_inr": int(pred_lakhs * 100000),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
