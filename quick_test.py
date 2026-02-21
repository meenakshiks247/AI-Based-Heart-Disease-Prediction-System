"""Quick single-sample prediction test for all saved models."""
import joblib, pandas as pd, numpy as np, os, sklearn, warnings

warnings.filterwarnings("ignore")
print("sklearn", sklearn.__version__)

df = pd.read_csv("data/heart_cleaned.csv")
sample = df.drop(columns=["target"]).iloc[[0]]
true_label = df["target"].iloc[0]

print(f"Sample (row 0):  true_label = {true_label}")
print(f"Features: {dict(sample.iloc[0])}\n")

header = f"{'Model':<32} {'Status':<6} {'Pred':<6} {'Prob(0)':<10} {'Prob(1)':<10}"
print(header)
print("-" * len(header))

models_dir = "ml/models"
for fname in sorted(os.listdir(models_dir)):
    if fname.endswith(".joblib"):
        path = os.path.join(models_dir, fname)
        try:
            model = joblib.load(path)
            pred = model.predict(sample)[0]
            prob_str_0, prob_str_1 = "-", "-"
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(sample)[0]
                prob_str_0 = f"{prob[0]:.4f}"
                prob_str_1 = f"{prob[1]:.4f}"
            print(f"  {fname:<30} {'OK':<6} {pred:<6} {prob_str_0:<10} {prob_str_1:<10}")
        except Exception as e:
            print(f"  {fname:<30} {'FAIL':<6} {str(e)[:60]}")

print(f"\nTrue label = {true_label}  (0 = No heart disease, 1 = Heart disease)")
