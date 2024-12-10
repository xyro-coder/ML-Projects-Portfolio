# Heart Disease Prediction Project

## Overview
This project implements various machine learning models to predict heart disease using patient health data. The implementation includes data preprocessing, feature engineering, exploratory data analysis (EDA), and model evaluation using multiple classification algorithms.

## Features
- Data preprocessing and cleaning
- Feature engineering with health-specific metrics
- Comprehensive EDA with visualizations
- Multiple classification models:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Support Vector Classification (SVC)
- Model evaluation using accuracy and ROC-AUC scores

## Dependencies
- pandas
- seaborn
- matplotlib
- numpy
- scikit-learn

## Dataset
The project uses a heart disease dataset with the following features:
- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope of Peak Exercise ST Segment
- Number of Major Vessels (ca)
- Thalassemia (thal)
- Target (heart disease presence)

## Feature Engineering
The project includes several engineered features:
1. Blood Pressure Categories:
   - Healthy_Trestbps
   - Elevated_Trestbps
   - Stage_1_Hypertension
   - Stage_2_Hypertension

2. Age-related Combinations:
   - Age_Chol
   - Age_Thalach
   - Age_Trestbps

3. Cholesterol Categories:
   - Normal_Chol
   - Borderline_High_Chol
   - High_Chol

## Model Selection
The project implements a comparative analysis of multiple classification models:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Classifier (SVC)

Each model is evaluated using:
- Accuracy Score
- ROC-AUC Score
- Average of both scores for final comparison

## Usage
1. Ensure all dependencies are installed:
```bash
pip install pandas seaborn matplotlib numpy scikit-learn
```

2. Load and preprocess the data:
```python
df = pd.read_csv("heart.csv")
```

3. Run the feature engineering and model selection:
```python
# Run main accuracy comparison
best_classifier, best_accuracy = main_accuracy()
```

## Visualizations
The project includes various visualizations:
- Distribution plots for numerical features
- Pair plots for feature relationships
- Correlation heatmaps
- Feature importance analysis using mutual information

## Data Split and Scaling
- Train-validation split: 70-30
- Standard scaling applied to features
- Random state set to 42 for reproducibility
