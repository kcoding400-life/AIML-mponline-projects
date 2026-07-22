"""
Trains a car resale-value predictor on car_data.csv and saves a single
scikit-learn Pipeline (preprocessing + model) to model/car_price_model.pkl
so the Flask app can load and use it directly.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("car_data.csv")

# Feature engineering: car age is more useful to the model than raw year
CURRENT_YEAR = 2026
df["Car_Age"] = CURRENT_YEAR - df["Year"]

numeric_features = ["Present_Price", "Kms_Driven", "Car_Age", "Owner"]
categorical_features = ["Brand", "Fuel_Type", "Seller_Type", "Transmission"]

X = df[numeric_features + categorical_features]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE:  {mae:.3f} lakhs")
print(f"R2:   {r2:.4f}")

joblib.dump(pipeline, "model/car_price_model.pkl")

# Save metadata used by the web form (dropdown choices, ranges, metrics)
metadata = {
    "brands": sorted(df["Brand"].unique().tolist()),
    "fuel_types": sorted(df["Fuel_Type"].unique().tolist()),
    "seller_types": sorted(df["Seller_Type"].unique().tolist()),
    "transmissions": sorted(df["Transmission"].unique().tolist()),
    "year_min": int(df["Year"].min()),
    "year_max": CURRENT_YEAR,
    "metrics": {"mae_lakhs": round(mae, 3), "r2": round(r2, 4)},
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model/car_price_model.pkl and model/metadata.json")
