# Personal Productivity Pattern Analyzer

## About the Project
The **Personal Productivity Pattern Analyzer** is a web-based application designed to analyze students' daily activities and understand how their routines relate to productivity.

It uses factors such as study hours, sleep duration, phone usage, social media usage, exercise, and academic activities to predict productivity scores and explore behavioral patterns.

## Objectives
- Analyze daily activity data and identify factors associated with productivity.
- Predict productivity scores and categorize productivity levels.
- Discover similar behavioral patterns among students.
- Identify unusual activity patterns.
- Provide a user-friendly dashboard for tracking productivity and personal progress.

## Key Features
- **Productivity Analysis:** View a predicted productivity score and level.
- **Daily Check-in:** Record daily activity information.
- **Behavior Patterns:** Explore similarities in activity routines.
- **Unusual Activity Detection:** Flag records that differ from common patterns.
- **Productivity History:** Review previous check-ins.
- **Goals and Insights:** Track personal goals and review routine-related insights.

## Technologies Used
- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Analysis and Modeling:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Development Tools:** Visual Studio Code, Git, GitHub

## Project Workflow
1. Load and inspect the student activity dataset.
2. Clean and preprocess the data.
3. Train and compare models for productivity-score prediction.
4. Explore behavioral groups and detect unusual records.
5. Connect the analysis to a web interface through an API.
6. Display predictions and activity insights through the dashboard.

## Dataset
The project uses structured student activity data containing lifestyle, study, and academic variables. The training dataset includes the target variable `productivity_score`, while the test dataset is used for prediction-related tasks.

## Running the Project

### 1. Clone the repository
```bash
git clone https://github.com/tejsree2007-design/ML-Project_Personal-Productivity-Pattern-Analyzer-.git
cd ML-Project_Personal-Productivity-Pattern-Analyzer-
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```

On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the application
Run the following command from the project root:
```bash
python -m uvicorn api.main:app --reload
```

### 5. Open the application
Visit:
- Application: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

**Note:** Train the models using the project's training script before starting the application if the required model files are not already present.

## Expected Outcome
A web-based productivity analysis prototype that combines daily activity tracking, productivity-score prediction, behavioral pattern exploration, and personal progress monitoring.

## Future Enhancements
- Database-backed storage and secure user authentication.
- Improved personalized recommendations.
- Long-term productivity trends and weekly reports.
- Deployment for access across devices.

## Project Team
Developed as a student project under the Computer Science and Engineering program.
