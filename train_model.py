import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\Project\Crop_Prediction-main\Crop_recommendation.csv")

# Features and labels
X = data.drop("label", axis=1)
y = data["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)

# Save all pkl files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ model.pkl, scaler.pkl, and label_encoder.pkl created successfully!")
