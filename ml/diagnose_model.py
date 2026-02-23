"""Quick diagnostic: check label mapping, scaling, feature order, and a test prediction."""
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"
TRAIN_PATH = Path(__file__).parent / "data" / "train.csv"

print("=" * 60)
print("1. MODEL INSPECTION")
print("=" * 60)
model = joblib.load(MODEL_PATH)
print(f"  Type          : {type(model).__name__}")
print(f"  Steps         : {[(n, type(s).__name__) for n, s in model.steps]}")

pp = model.named_steps["preprocessor"]
scaler = pp.named_transformers_["num"]["scaler"]
features = pp.feature_names_in_.tolist()
print(f"  Feature order : {features}")
print(f"  Scaler mean   : {scaler.mean_}")
print(f"  Scaler scale  : {scaler.scale_}")
print(f"  model.classes_: {model.classes_}")

print()
print("=" * 60)
print("2. TARGET LABEL CHECK")
print("=" * 60)
df = pd.read_csv(TRAIN_PATH)
print(f"  target unique : {sorted(df['target'].unique())}")
print(f"  target counts :\n{df['target'].value_counts().to_string()}")

print()
print("=" * 60)
print("3. CATEGORICAL VALUE RANGES (training data)")
print("=" * 60)
for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
    print(f"  {c:10s} : {sorted(df[c].unique())}")

print()
print("=" * 60)
print("4. DIAGNOSTIC PREDICTIONS")
print("=" * 60)

# A clearly healthy-looking patient
healthy = pd.DataFrame(
    [[29, 0, 0, 110, 180, 0, 0, 180, 0, 0.0, 2, 0, 2]],
    columns=features,
)
# A clearly at-risk patient
risky = pd.DataFrame(
    [[65, 1, 3, 180, 350, 1, 2, 100, 1, 3.5, 2, 3, 3]],
    columns=features,
)
# Default form values from the frontend
default = pd.DataFrame(
    [[54, 1, 0, 130, 250, 0, 0, 150, 0, 1.0, 1, 0, 2]],
    columns=features,
)

for label, sample in [("HEALTHY", healthy), ("RISKY", risky), ("DEFAULT", default)]:
    proba = model.predict_proba(sample)[0]
    pred = model.predict(sample)[0]
    print(f"  {label:8s} => pred={pred}, proba_class0={proba[0]:.4f}, proba_class1={proba[1]:.4f}")

print()
print("  classes_[0] = 'no disease', classes_[1] = 'disease'")
print("  predict_proba(X)[0][1] = probability of DISEASE (class 1)")
