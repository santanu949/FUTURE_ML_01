from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import sys
import os

# Ensure project root is in path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.engine import ModelEngine
from src.data.processor import DataProcessor

app = FastAPI(title="Future Forecast API")

@app.get("/")
def read_root():
    return {"message": "Welcome to Future Forecast API. Use /predict for sales projections."}

@app.post("/predict")
async def predict_sales(file: UploadFile = File(...), horizon: int = 30):
    """
    Upload a CSV with 'ds' and 'y' columns to get sales predictions.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Process
    df = DataProcessor.validate_data(df, 'ds', 'y')
    
    # Train (Simplified for API)
    engine = ModelEngine(df)
    m = engine.train_best_prophet() # Uses default or last tuned params
    
    # Forecast
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    
    # Return as JSON
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).to_dict(orient='records')
    return {"model": "prophet", "horizon": horizon, "predictions": results}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
