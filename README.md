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
