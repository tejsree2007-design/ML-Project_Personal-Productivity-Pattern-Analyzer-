from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cleaned_train.csv"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

target = "productivity_score"

X = df.drop(columns=[target])
y = df[target]

numeric_columns = X.select_dtypes(include=["number"]).columns

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_processed = preprocessor.fit_transform(X[numeric_columns])

# --------------------------------------------------
# REGRESSION EVALUATION
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)

regression_models = {
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

regression_results = {}

for name, model in regression_models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    regression_results[name] = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(
            y_test,
            predictions
        ) ** 0.5,
        "R2": r2_score(y_test, predictions)
    }

# Cross-validation
cv_results = {}

for name, model in regression_models.items():

    scores = cross_val_score(
        model,
        X_processed,
        y,
        cv=5,
        scoring="r2"
    )

    cv_results[name] = {
        "CV_R2_mean": scores.mean(),
        "CV_R2_std": scores.std()
    }

# --------------------------------------------------
# RIDGE HYPERPARAMETER TUNING
# --------------------------------------------------

ridge_grid = GridSearchCV(
    Ridge(),
    {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="neg_root_mean_squared_error"
)

ridge_grid.fit(X_processed, y)

ridge_tuning = {
    "best_alpha": ridge_grid.best_params_["alpha"],
    "best_cv_RMSE": -ridge_grid.best_score_
}

# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

def productivity_level(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    return "High"


y_class = y.apply(productivity_level)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_processed,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

classification_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

classification_results = {}

# Use proper classifiers
from sklearn.ensemble import RandomForestClassifier

classification_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

# Correct Gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier

classification_models["Gradient Boosting"] = GradientBoostingClassifier(
    random_state=42
)

for name, model in classification_models.items():

    model.fit(X_train_c, y_train_c)
    predictions = model.predict(X_test_c)

    classification_results[name] = {
        "Accuracy": accuracy_score(
            y_test_c,
            predictions
        ),
        "Precision": precision_score(
            y_test_c,
            predictions,
            average="weighted"
        ),
        "Recall": recall_score(
            y_test_c,
            predictions,
            average="weighted"
        ),
        "F1": f1_score(
            y_test_c,
            predictions,
            average="weighted"
        )
    }

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

best_classifier = classification_models["Gradient Boosting"]

best_classifier.fit(X_train_c, y_train_c)

ConfusionMatrixDisplay.from_estimator(
    best_classifier,
    X_test_c,
    y_test_c
)

plt.title("Productivity Level Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT / "classification_confusion_matrix.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# LEARNING CURVE
# --------------------------------------------------

train_sizes, train_scores, validation_scores = learning_curve(
    Ridge(alpha=1.0),
    X_processed,
    y,
    cv=5,
    scoring="r2",
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0]
)

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    train_scores.mean(axis=1),
    label="Training R²"
)

plt.plot(
    train_sizes,
    validation_scores.mean(axis=1),
    label="Validation R²"
)

plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.title("Ridge Learning Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT / "ridge_learning_curve.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results = {
    "regression": regression_results,
    "cross_validation": cv_results,
    "ridge_tuning": ridge_tuning,
    "classification": classification_results
}

with open(
    OUTPUT / "evaluation_results.json",
    "w"
) as file:
    json.dump(
        results,
        file,
        indent=4
    )

print("\nM5 Evaluation Completed.")
print("\nRegression:")
print(json.dumps(regression_results, indent=4))

print("\nCross-validation:")
print(json.dumps(cv_results, indent=4))

print("\nRidge tuning:")
print(json.dumps(ridge_tuning, indent=4))

print("\nClassification:")
print(json.dumps(classification_results, indent=4))

print("\nGenerated:")
print("- evaluation_results.json")
print("- classification_confusion_matrix.png")
print("- ridge_learning_curve.png")