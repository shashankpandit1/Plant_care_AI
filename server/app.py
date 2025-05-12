from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from bson import ObjectId
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import json

load_dotenv()

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "recommendation_model", "plant_recommender.pkl")
encoder_path = os.path.join(BASE_DIR, "recommendation_model", "label_encoder.pkl")
# Load the model and label encoder
# model = joblib.load(model_path)
# label_encoder = joblib.load(encoder_path)
recommender_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)


class SensorReading(BaseModel):
    temperature: float
    humidity: float
    light: float


# Allow CORS (if ESP32 or a browser front‐end will call you)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["plant_monitoring"]
sensor_collection    = db["sensor_data"]
threshold_collection = db["thresholds"]

# Load the plant dataset from CSV once on startup
plant_df = pd.read_csv("plant_environment_data.csv")
class SensorData(BaseModel):
    temperature: float
    humidity: float
    light: float
    
class Threshold(BaseModel):
    min_soil_moisture: float
    max_soil_moisture: float


# Load model and class names
with open("model/class_names.json") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model/plant_classifier.pth", map_location=device))
model.eval().to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ───────────────────────────────────────────────
# Sensor Data Routes (ESP32 <-> MongoDB)
# ───────────────────────────────────────────────

@app.post("/sensor/")
def receive_sensor_data(data: SensorData):
    entry = data.dict()
    target_id = ObjectId("6814319834cc9c74c409f4a1")

    result = sensor_collection.update_one(
        {"_id": target_id},
        {"$set": entry}
    )

    if result.matched_count:
        return {"message": "Sensor data updated successfully"}
    else:
        return {"message": "Document not found. No update performed."}


@app.get("/thresholds/", response_model=List[float])
def get_avg_thresholds():
   
    docs = list(threshold_collection.find().limit(3))
    if len(docs) < 3:
        raise HTTPException(404, detail="Not enough threshold docs")
    return [
        round((doc["min_soil_moisture"] + doc["max_soil_moisture"])/2, 2)
        for doc in docs
    ]
# @app.get("/thresholds/", response_model=List[float])
# def get_avg_thresholds():
#     # Get exactly 2 docs now
#     docs = list(threshold_collection.find().limit(2))
#     if len(docs) < 2:
#         raise HTTPException(404, detail="Not enough threshold docs")
#     return [
#         round((doc["min_soil_moisture"] + doc["max_soil_moisture"]) / 2, 2)
#         for doc in docs
#     ]
# ───────────────────────────────────────────────
# Plant Dataset API Routes (Frontend <-> MongoDB)
# ───────────────────────────────────────────────
@app.get("/sensor-data/")
def get_sensor_data():
    doc = sensor_collection.find_one({"_id": ObjectId("6814319834cc9c74c409f4a1")})
    if not doc:
        raise HTTPException(status_code=404, detail="Sensor data not found")
    doc.pop("_id", None)
    return doc


@app.get("/plants/")
def list_plants():
    """Return unique plant names from CSV for dropdown selection"""
    # plants = plant_df["Plant Name"].drop_duplicates().tolist()
    return plant_df[['Plant Name']].to_dict(orient='records')

@app.get("/plant_threshold/")
def get_plant_threshold(name: str = Query(..., description="Exact plant name")):
    """Return min/max soil moisture thresholds for the selected plant"""
    matches = plant_df[plant_df["Plant Name"] == name]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Plant not found")

    plant = matches.iloc[0]
    return {
        "min_soil_moisture": int(plant["Min Soil Moisture"]),
        "max_soil_moisture": int(plant["Max Soil Moisture"])
    }

# 
@app.post("/watering-thresholds/")
def save_thresholds(thresholds: List[Threshold]):
    if len(thresholds) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 thresholds required")

    existing_docs = list(threshold_collection.find().limit(3))
    if len(existing_docs) != 3:
        raise HTTPException(status_code=404, detail="Exactly 3 existing threshold documents required")

    updated_ids = []
    for doc, new_threshold in zip(existing_docs, thresholds):
        result = threshold_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": new_threshold.dict()}
        )
        updated_ids.append(str(doc["_id"]))

    return {"updated_ids": updated_ids}

# @app.post("/recommend-plants/")
# def recommend_plants(reading: SensorReading):
    

#     input_df = pd.DataFrame([{
#         "temperature": reading.temperature,
#         "humidity": reading.humidity,
#         "light": reading.light,
#     }])
#     input_df.rename(columns={
#     'humidity': 'Humidity',
#     'light': 'Light',
#     'temperature': 'Temperature'
#     }, inplace=True)

    
#     predictions = model.predict_proba(input_df)[0]
#     top_indices = predictions.argsort()[-3:][::-1]
#     top_plants = label_encoder.inverse_transform(top_indices)

#     return {"recommended_plants": top_plants.tolist()}

@app.post("/recommend-plants/")
def recommend_plants(reading: SensorReading):
    input_df = pd.DataFrame([{
        "temperature": reading.temperature,
        "humidity": reading.humidity,
        "light": reading.light,
    }])
    
    input_df.rename(columns={
        'humidity': 'Humidity',
        'light': 'Light',
        'temperature': 'Temperature'
    }, inplace=True)

    predictions = recommender_model.predict_proba(input_df)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_plants = label_encoder.inverse_transform(top_indices)

    return {"recommended_plants": top_plants.tolist()}


@app.post("/identify-plant/")
async def identify_plant(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        plant_name = class_names[predicted.item()]

    return {"plant_name": plant_name}
