import json
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURES, MODEL_DIR, OUTPUT_DIR, TARGET, TRAIN_FILE

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(TRAIN_FILE)
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "random_forest": RandomForestRegressor(
        n_estimators=250, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}

results = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results[name] = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)),
        "CV_R2_mean": float(cross_val_score(model, X_train, y_train, cv=cv, scoring="r2").mean()),
    }

    joblib.dump(model, MODEL_DIR / f"{name}_regression.joblib")

with open(OUTPUT_DIR / "regression_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

best_name = min(results, key=lambda n: results[n]["RMSE"])
print("Regression results:")
for name, metrics in results.items():
    print(name, metrics)
print("Best model by RMSE:", best_name)
