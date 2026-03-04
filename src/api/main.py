from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
from pathlib import Path

# Load model
MODEL_PATH = Path("models/player_value_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Get feature names from training data
df = pd.read_csv("data/processed/players_processed.csv")
FEATURE_COLUMNS = [col for col in df.columns if col != "value_eur"]

# Initialise FastAPI app
app = FastAPI(
    title="Footballer Value Predictor",
    description="Predict a footballer's market value based on their stats",
    version="1.0.0"
)

# Define input schema
class PlayerInput(BaseModel):
    age: int
    overall: int
    potential: int
    wage_eur: float
    international_reputation: int
    weak_foot: int
    skill_moves: int
    pace: float
    shooting: float
    passing: float
    dribbling: float
    defending: float
    physic: float
    preferred_foot: int  # 1 = Right, 0 = Left
    years_on_contract: float
    position_ATT: int
    position_DEF: int
    position_GK: int
    position_MID: int
    attack_work_rate_High: int
    attack_work_rate_Low: int
    attack_work_rate_Medium: int
    defense_work_rate_High: int
    defense_work_rate_Low: int
    defense_work_rate_Medium: int

@app.get("/")
def root():
    return {"message": "Footballer Value Predictor API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(player: PlayerInput):
    # Convert input to DataFrame
    input_data = pd.DataFrame([player.model_dump()])
    
    # Ensure column order matches training data
    input_data = input_data[FEATURE_COLUMNS]
    
    # Predict (returns log-transformed value)
    log_prediction = model.predict(input_data)[0]
    
    # Reverse log transformation to get euros
    predicted_value = np.expm1(log_prediction)
    
    return {
        "predicted_value_eur": round(float(predicted_value), 2),
        "predicted_value_millions": round(float(predicted_value) / 1_000_000, 2)
    }