from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "cleaned_train.csv"
TEST_FILE = DATA_DIR / "cleaned_test.csv"
TARGET = "productivity_score"

FEATURES = [
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "social_media_hours",
    "youtube_hours",
    "gaming_hours",
    "exercise_minutes",
    "breaks_per_day",
]

LOW_THRESHOLD = 40.0
HIGH_THRESHOLD = 70.0

def productivity_level(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "Low"
    if score < HIGH_THRESHOLD:
        return "Medium"
    return "High"
