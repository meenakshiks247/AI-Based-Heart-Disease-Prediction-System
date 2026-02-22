## Project Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection is critical for effective treatment and lifestyle intervention. 
This project aims to assist healthcare professionals in early diagnosis by identifying high-risk patients through data analysis. It utilizes a 14-feature clinical dataset to classify patients into "At Risk" or "Healthy" categories.

## Tech Stack
- **Language:**  Python 3.x
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (for baseline models)
- **Deep Learning:** TensorFlow / Keras (for the final predictive model)

### Key Objectives:
* **Predictive Modeling:** Build a binary classifier to categorize patients into 'Healthy' or 'At Risk'.
* **Clinical Insight:** Identify the most significant risk factors (features) contributing to heart disease using correlation analysis.
* **Scalable AI:** Utilize a Deep Learning approach with **TensorFlow** to improve upon traditional baseline machine learning models.
  
## Dataset
- The project uses the UCI Heart Disease Dataset (Cleveland version).
- **Link:** https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Features:** 13 clinical attributes (Age, Sex, Chest Pain Type, Cholesterol, Resting BP, etc.).
- **Target:** Binary classification (1 = Presence of disease, 0 = Absence).

## Solution Approach

My approach follows a structured machine learning pipeline:
- **Exploratory Data Analysis (EDA):** Visualizing correlations between clinical features and heart disease risk.
- **Data Preprocessing:** Handling categorical encoding and feature scaling using Scikit-learn.
- **Model Development:** Establishing a baseline with Scikit-learn and developing a Deep Learning model using TensorFlow.
- **Evaluation:** Assessing the model using Accuracy, Precision, and Recall metrics.

## Data Preprocessing & Feature Engineering

This phase focuses on transforming raw clinical data into a high-quality format optimized for Deep Learning (TensorFlow).
1. **Outlier Management (Clinical Trimming)**
- Medical data often contains extreme values that can disproportionately influence model weights.
- Technique: Interquartile Range (IQR) Capping.
- Application: Features like chol (Cholesterol) and trestbps (Resting Blood Pressure) showed significant right-skewed outliers.
- Action: Values exceeding the Upper Fence ($Q3 + 1.5 \times IQR$) were capped at the threshold rather than removed, preserving the dataset size ($N=1,025$) while stabilizing the variance.
2. **Feature Correlation & Selection**
- To reduce model complexity and prevent Multicollinearity, I performed a statistical audit of the feature set.
- Observation: Features like cp (Chest Pain Type) and thalach (Max Heart Rate) showed the highest absolute correlation with heart disease risk.
- Redundancy Check: Verified that no two independent variables were perfectly correlated ($r > 0.9$), ensuring the model learns unique patterns from each feature.
3. **Data Integrity & Checkpointing**
- Handling Nulls: Confirmed 0 missing values; no imputation required.
- Output: The processed data is exported to data/heart_cleaned.csv to ensure a consistent baseline for both Scikit-Learn and TensorFlow experiments.
4. **Duplicate Value Removal**
- Exact duplicate records are detected and removed before safe training.
- Main utility: `ml/deduplicate.py` (supports `keep-first`, `keep-random`, and `keep-most-representative` strategies).
- Deduplicated output: `ml/data/heart_cleaned_dedup.csv`.

## Model training (Day 3)

This stage trains and compares 7 baseline ML models on the cleaned dataset:
- Logistic Regression
- Random Forest
- SVM
- Naive Bayes
- Decision Tree
- LightGBM
- XGBoost

Saved outputs:
- Trained model artifacts: `ml/models/*.joblib`
- Model metrics table: `ml/models/model_results.csv`

Run command:
```bash
python ml/run_training.py
```

Note: `compare_models.py` selects the top model and saves it to `ml/models/best_model.joblib`.

## Safe Training Pipeline

To reduce leakage/overfitting risk, the project also includes a safe data + training flow:

1. Deduplicate:
```bash
python ml/deduplicate.py --strategy keep-first
```

2. Stratified split:
```bash
python ml/safe_split.py
```

3. Train with CV on train split:
```bash
python ml/train_models_safe.py
```

4. Select best safe model:
```bash
python ml/compare_models.py
```

Safe outputs:
- CV results: `ml/models/model_results_safe.csv`
- Saved safe models: `ml/models/*_safe.joblib`
- Selected best model: `ml/models/best_model.joblib`
- Best-model metadata: `ml/models/best_model_info.json`

## Data Quality & Leakage Utilities

- Duplicate inspection: `python ml/check_duplicates.py [--drop]`
- Leakage scan per feature: `python ml/leakage_scan.py`
- Overfitting verification: `python ml/verify_overfitting.py --data ml/data/heart_cleaned.csv`

## Backend Integration

- Model loader: `backend/app/ml/load_model.py`
- API routes: `backend/app/api/models_api.py`
  - `GET /api/models/`
  - `GET /api/models/best`
