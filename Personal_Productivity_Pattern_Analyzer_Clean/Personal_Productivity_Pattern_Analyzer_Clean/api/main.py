from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ridge_regression.joblib"
app = FastAPI(title="Personal Productivity Pattern Analyzer", version="1.0")

class PredictionRequest(BaseModel):
    study_hours_per_day: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=24)
    phone_usage_hours: float = Field(..., ge=0, le=24)
    social_media_hours: float = Field(..., ge=0, le=24)
    youtube_hours: float = Field(..., ge=0, le=24)
    gaming_hours: float = Field(..., ge=0, le=24)
    exercise_minutes: float = Field(..., ge=0)
    breaks_per_day: float = Field(..., ge=0)

@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL.exists()}

@app.post("/predict")
def predict(req: PredictionRequest):
    if not MODEL.exists():
        return {"error": "Model not found. Run python src/train.py first."}

    X = pd.DataFrame([req.model_dump()])
    model = joblib.load(MODEL)
    score = float(model.predict(X)[0])
    score = max(0.0, min(100.0, score))

    if score < 40:
        level = "Low"
    elif score < 70:
        level = "Medium"
    else:
        level = "High"

    return {
        "productivity_score": round(score, 2),
        "productivity_level": level
    }
