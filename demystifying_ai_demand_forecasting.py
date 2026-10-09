"""DataCamp: Demystifying AI — Explaining Demand Forecasting Models.

Run in the DataCamp project folder, alongside its ``data/`` directory,
or paste the whole script into a Python 3.10 notebook cell.

Requires: numpy, pandas, shap, scikit-learn, joblib.
The two pretrained .pkl files and four CSVs come with the DataCamp project.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# 1. Load the project datasets and pretrained models.
# -----------------------------------------------------------------------------
np.random.seed(42)
DATA_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
            else Path.cwd()) / "data"

required_files = [
    "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
    "model.pkl", "knn_model.pkl",
]
missing = [name for name in required_files if not (DATA_DIR / name).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing project files in {DATA_DIR}: {missing}. "
        "Run this from the DataCamp project directory with its data/ folder."
    )

X_train_all = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_all = pd.read_csv(DATA_DIR / "X_test.csv")
X_train = X_train_all.sample(min(500, len(X_train_all)), random_state=42)
X_test = X_test_all.sample(min(500, len(X_test_all)), random_state=42)
y_train = pd.read_csv(DATA_DIR / "y_train.csv")["nextmonth__home_decor"]
y_test = pd.read_csv(DATA_DIR / "y_test.csv")["nextmonth__home_decor"]
y_train = y_train.sample(min(500, len(y_train)), random_state=42)
y_test = y_test.sample(min(500, len(y_test)), random_state=42)
model = joblib.load(DATA_DIR / "model.pkl")
knn_model = joblib.load(DATA_DIR / "knn_model.pkl")

# -----------------------------------------------------------------------------
# 2. Task 1: Name and import the game-theory-based explainability library.
# -----------------------------------------------------------------------------
xai = "shap"  # String required by DataCamp; the library was imported above.

# -----------------------------------------------------------------------------
# 3. Task 2: Explain the RandomForestRegressor and find its top five features.
# -----------------------------------------------------------------------------
rf_explainer = shap.TreeExplainer(model)
rf_shap_values = rf_explainer.shap_values(X_test)
rf_importance = np.abs(np.asarray(rf_shap_values)).mean(axis=0)

rf_features = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": rf_importance,
})
top_feats = rf_features.sort_values(
    by="Importance", ascending=False
).head(5).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 4. Task 3: Explain the KNN model on 50 test rows.
#    Kernel SHAP is computationally expensive. Use a small background dataset
#    and a limited sample budget so this exercise remains reasonably fast.
# -----------------------------------------------------------------------------
background = X_train.sample(min(10, len(X_train)), random_state=42)
knn_test = X_test.sample(min(50, len(X_test)), random_state=42)
knn_explainer = shap.KernelExplainer(knn_model.predict, background)
knn_shap_values = knn_explainer.shap_values(
    knn_test, nsamples=120, silent=True
)
knn_importance = np.abs(np.asarray(knn_shap_values)).mean(axis=0)

knn_features = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": knn_importance,
})
knn_top_feats = knn_features.sort_values(
    by="Importance", ascending=False
).head(5).reset_index(drop=True)

# Compare scores for MATCHING features in matching column order.
# Using the fully aligned vectors avoids comparing unrelated features merely
# because they happen to occupy the same rank in different top-five tables.
consistency = float(round(
    cosine_similarity([rf_importance], [knn_importance])[0][0], 2
))

# -----------------------------------------------------------------------------
# 5. Task 4: Interpret the consistency of the explanations.
#    0.80 is a practical example threshold, NOT a prescribed DataCamp cutoff.
# -----------------------------------------------------------------------------
reliable = "yes" if consistency >= 0.80 else "no"

# -----------------------------------------------------------------------------
# 6. Display the required results.
# -----------------------------------------------------------------------------
print("XAI library:", xai)
print("\nRandom forest — top five features:\n", top_feats.to_string(index=False))
print("\nKNN — top five features:\n", knn_top_feats.to_string(index=False))
print("\nCosine similarity (consistency):", consistency)
print("Feature-importance explanations consistent?", reliable)
print("\nDataCamp variables ready: xai, top_feats, consistency, reliable")
