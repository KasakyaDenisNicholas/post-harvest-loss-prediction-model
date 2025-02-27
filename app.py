from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
try:
    model = joblib.load("decision_tree_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

# Define request body structure
class PostHarvestData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(data: PostHarvestData):
    input_data = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}

@app.get("/")
def home():
    return {"message": "Post-Harvest Loss Prediction API is running!"}
