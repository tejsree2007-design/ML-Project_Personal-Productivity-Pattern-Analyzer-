# Personal Productivity Pattern Analyzer

Clean-feature version of the project.

## Dataset
The raw folder contains the cleaned training and testing CSV files.

Features:
- study_hours_per_day
- sleep_hours
- phone_usage_hours
- social_media_hours
- youtube_hours
- gaming_hours
- exercise_minutes
- breaks_per_day

Target:
- productivity_score

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

```powershell
python src/eda.py
python src/train.py
python src/classification.py
python src/clustering.py
python src/anomaly_detection.py
```

## Start the application

```powershell
python -m uvicorn api.main:app --reload
```

Open:
http://127.0.0.1:8000

See COURSE_OUTCOME_COVERAGE.md for the M1-M6 mapping.
