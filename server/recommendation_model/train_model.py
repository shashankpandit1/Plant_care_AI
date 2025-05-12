import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("../plant_environment_data.csv")

# Calculate averages for features
df["Temperature"] = (df["Min Temp"] + df["Max Temp"]) / 2
df["Humidity"] = (df["Min Humidity"] + df["Max Humidity"]) / 2
df["Light"] = (df["Min Light"] + df["Max Light"]) / 2
# df["soil_moisture"] = (df["Min Soil Moisture"] + df["Max Soil Moisture"]) / 2

# Features & Target
X = df[["Temperature", "Humidity", "Light"]]
y = df["Plant Name"]

# Encode plant names
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save model and encoder
joblib.dump(model, "plant_recommender.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved as plant_recommender.pkl")
