import json
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURES, MODEL_DIR, OUTPUT_DIR, TARGET, TRAIN_FILE, productivity_level

df = pd.read_csv(TRAIN_FILE)
X = df[FEATURES]
y = df[TARGET].apply(productivity_level)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    results[name] = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
    }
    joblib.dump(model, MODEL_DIR / f"{name}_classifier.joblib")

with open(OUTPUT_DIR / "classification_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
