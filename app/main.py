from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict
import pandas as pd

app = FastAPI()

class Request(BaseModel):
    age: int
    sex: int
    cholesterol: int
    heart_rate: int
    diabetes: int
    family_history: int
    smoking: int
    obesity: int
    alcohol_consumption: int
    exercise_hours_per_week: float
    diet: int
    previous_heart_problems: int
    medication_use: int
    stress_level: int
    sedentary_hours_per_day: float
    bmi: float
    triglycerides: int
    physical_activity_days_per_week: float
    sleep_hours_per_day: int
    continent: int
    hemisphere: int
    bp_systolic: int
    bp_diastolic: int
    
class Response(BaseModel):
    prediction: float

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.post("/predict", response_model=Response)
def predict_handler(data: Request):
    input_data = pd.DataFrame([data.dict()])
    prediction = predict(input_data)
    return {"prediction": prediction}