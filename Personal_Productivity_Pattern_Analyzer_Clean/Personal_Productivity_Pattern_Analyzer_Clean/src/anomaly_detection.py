import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import FEATURES, MODEL_DIR, OUTPUT_DIR, TRAIN_FILE

df = pd.read_csv(TRAIN_FILE)
X = df[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X_scaled)

result = pd.DataFrame({
    "anomaly": predictions
})

result["status"] = result["anomaly"].map({
    1: "Normal",
    -1: "Unusual"
})

result.to_csv(
    OUTPUT_DIR / "anomaly_results.csv",
    index=False
)

joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.joblib")
joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")

print("Anomaly detection completed.")
print(result["status"].value_counts())