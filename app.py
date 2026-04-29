from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Enable CORS so your Netlify site can talk to Hugging Face
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your files
model = joblib.load("model.pkl")
le_dict = joblib.load("encoders.pkl")

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
        cat_enc = le_dict['Category'].transform([data.category])[0]
        weath_enc = le_dict['Weather'].transform([data.weather])[0]
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
