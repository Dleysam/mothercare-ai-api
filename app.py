from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Enable CORS for Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the single package
# Make sure the filename matches exactly what you upload to GitHub
pkg = joblib.load("mothercare_model_v1.pkl")

# 2. Extract components
model = pkg['model']
le_dict = {
    'Category': pkg['le_cat'],
    'Weather': pkg['le_weather']
}

class PredictRequest(BaseModel):
    category: str
    weather: str
    day_final: int
    start_hour: int = 10
    month: int
    day: int
    payday_bin: int
    shelf_status: int = 1
    temp: float = 28.5
    humidity: float = 65.0
    precip: float = 0.0

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # Encoding words to numbers
        cat_enc = le_dict['Category'].transform([data.category])[0]
        weath_enc = le_dict['Weather'].transform([data.weather])[0]
        
        # The 11-feature order your model expects
        feature_order = [
            'Category_Code', 'Day_Final', 'Weather_Final', 'Start_Hour',
            'Month', 'Day', 'Payday_Bin', 'Shelf_Status',
            'External_Temp', 'External_Humidity', 'External_Precip'
        ]
        
        input_data = pd.DataFrame([{
            'Category_Code': cat_enc,
            'Day_Final': data.day_final,
            'Weather_Final': weath_enc,
            'Start_Hour': data.start_hour,
            'Month': data.month,
            'Day': data.day,
            'Payday_Bin': data.payday_bin,
            'Shelf_Status': data.shelf_status,
            'External_Temp': data.temp,
            'External_Humidity': data.humidity,
            'External_Precip': data.precip
        }])[feature_order]
        
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0]), "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
