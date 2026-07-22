"""
Generates a synthetic but realistic used-car dataset for training the
car value predictor. Mimics the structure of common used-car datasets
(e.g. Year, Present Price, Kms Driven, Fuel Type, Seller Type,
Transmission, Owner) with realistic depreciation relationships baked in,
plus a bit of random noise.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 3000

brands = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Tata",
          "Mahindra", "Volkswagen", "Renault", "Skoda"]
brand_base_price = {  # base "present price" (in lakhs) per brand, roughly
    "Maruti": 7.0, "Hyundai": 8.5, "Honda": 9.5, "Toyota": 14.0,
    "Ford": 9.0, "Tata": 7.5, "Mahindra": 11.0, "Volkswagen": 10.5,
    "Renault": 7.0, "Skoda": 12.0,
}

fuel_types = ["Petrol", "Diesel", "CNG"]
seller_types = ["Dealer", "Individual"]
transmissions = ["Manual", "Automatic"]

current_year = 2026

rows = []
for _ in range(N):
    brand = rng.choice(brands)
    base_price = brand_base_price[brand] * rng.uniform(0.85, 1.25)  # model variation
    year = int(rng.integers(2008, 2026))
    age = current_year - year

    kms_driven = int(max(500, rng.normal(15000 * max(age, 1), 8000)))
    fuel_type = rng.choice(fuel_types, p=[0.55, 0.35, 0.10])
    seller_type = rng.choice(seller_types, p=[0.65, 0.35])
    transmission = rng.choice(transmissions, p=[0.80, 0.20])
    owner = int(rng.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05]))

    present_price = round(base_price, 2)  # current showroom price of this model today

    # --- Depreciation model to derive a realistic selling price ---
    depreciation = 0.88 ** age  # ~12% value loss per year, compounding
    km_penalty = 1 - min(kms_driven / 300000, 0.35)  # up to 35% penalty for high kms
    fuel_adj = {"Diesel": 1.03, "Petrol": 1.0, "CNG": 0.93}[fuel_type]
    seller_adj = {"Dealer": 1.05, "Individual": 0.95}[seller_type]
    trans_adj = {"Automatic": 1.08, "Manual": 1.0}[transmission]
    owner_adj = 1 - owner * 0.06

    selling_price = (present_price * depreciation * km_penalty *
                      fuel_adj * seller_adj * trans_adj * owner_adj)
    selling_price *= rng.uniform(0.93, 1.07)  # market noise
    selling_price = max(0.15, round(selling_price, 2))

    rows.append({
        "Brand": brand,
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Selling_Price": selling_price,
    })

df = pd.DataFrame(rows)
df.to_csv("car_data.csv", index=False)
print(df.head())
print(f"\nGenerated {len(df)} rows -> car_data.csv")
print(df.describe())
