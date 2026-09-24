import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("🚀 Step 1: Loading & mapping dataset...")
raw_df = pd.read_csv("data/car_data.csv")

# 1. Cleaning null values in key columns
key_cols = ["Manufacturer", "Vehicle_type", "Price_in_thousands", "Engine_size", "Fuel_efficiency"]
raw_df = raw_df.dropna(subset=key_cols)

# 2. Year extract karo Latest_Launch column se
if "Latest_Launch" in raw_df.columns:
    raw_df["Year"] = pd.to_datetime(raw_df["Latest_Launch"], errors="coerce").dt.year.fillna(2012).astype(int)
else:
    raw_df["Year"] = 2015

# 3. Clean Standard DataFrame create karo
df = pd.DataFrame()
df["Brand"] = raw_df["Manufacturer"].astype(str).str.strip()
df["Body"] = raw_df["Vehicle_type"].astype(str).str.strip().str.lower()
df["Mileage"] = raw_df["Fuel_efficiency"].astype(float) * 10.0  # Normalized scale
df["EngineV"] = raw_df["Engine_size"].astype(float)
df["Engine Type"] = "Petrol"
df["Registration"] = "yes"
df["Year"] = raw_df["Year"]  # Bug fixed here
df["Model"] = raw_df["Model"].astype(str).str.strip()

# Price (in Thousands USD to approximate INR)
df["Price"] = raw_df["Price_in_thousands"].astype(float) * 85000.0

# 4. Outlier cleaning
df = df[(df["Price"] > 100000) & (df["EngineV"] <= 6.5) & (df["EngineV"] >= 0.8)]

# 5. Dynamic Dropdown Metadata create karo
brand_models = {}
for brand, sub_df in df.groupby("Brand"):
    brand_models[brand] = sorted(sub_df["Model"].unique().tolist())

meta = {
    "brands": sorted(df["Brand"].unique().tolist()),
    "bodies": sorted(df["Body"].unique().tolist()),
    "engine_types": ["Petrol", "Diesel", "Electric", "Hybrid"],
    "registrations": ["yes", "no"],
    "brand_models": brand_models,
    "year_min": int(df["Year"].min()),
    "year_max": int(df["Year"].max()),
    "mileage_max": float(df["Mileage"].max()),
}

with open("artifacts/car_meta.json", "w") as f:
    json.dump(meta, f, indent=4)
print("✅ Saved metadata to artifacts/car_meta.json")

# 6. Feature Matrix & Target Setup
features = ["Brand", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year", "Model"]
X = df[features]
y = np.log1p(df["Price"])

num_cols = ["Mileage", "EngineV", "Year"]
cat_cols = ["Brand", "Body", "Engine Type", "Registration", "Model"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1)),
    ]
)

# 7. Model Training & Evaluation
print("⏳ Training RandomForest Pipeline...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
print(f"🎯 Model R² Score: {r2_score(y_test, preds):.4f}")
print(f"💰 Mean Absolute Error: ₹{mean_absolute_error(np.expm1(y_test), np.expm1(preds)):,.2f}")

joblib.dump(pipeline, "artifacts/car_price_pipeline.pkl")
print("✅ Exported trained pipeline to artifacts/car_price_pipeline.pkl")