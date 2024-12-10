# California Housing Price Prediction Project

## Overview
This project implements a machine learning pipeline to predict housing prices in California using various regression models. The implementation includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation using multiple regression algorithms.

## Features
- Automated data downloading and extraction
- Comprehensive data preprocessing pipeline
- Detailed exploratory data analysis with visualizations
- Multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regression (SVR)
- Model optimization using GridSearchCV and RandomizedSearchCV
- Feature importance analysis

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- urllib
- tarfile
- os

## Dataset
The project uses the California Housing dataset with the following features:
- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- median_house_value
- ocean_proximity

## Data Preprocessing Pipeline
1. Missing Value Handling:
   - SimpleImputer with median strategy
2. Feature Engineering:
   - rooms_per_household
   - bedrooms_per_room
   - population_per_household
3. Categorical Data Processing:
   - OneHotEncoder for ocean_proximity
4. Numerical Data Scaling:
   - StandardScaler

## Model Selection and Evaluation
The project implements multiple regression models:
1. Linear Regression (baseline)
2. Decision Tree Regressor
3. Random Forest Regressor (with GridSearchCV optimization)
4. Support Vector Regression

Each model is evaluated using:
- Root Mean Square Error (RMSE)
- Cross-validation scores
- Feature importance analysis (for tree-based models)

## Usage

1. Clone the repository and install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Download and load the data:
```python
from housing_data import fetch_housing_data, load_housing_data

# Fetch data
fetch_housing_data()

# Load into pandas DataFrame
housing = load_housing_data()
```

3. Run the preprocessing pipeline:
```python
from sklearn.pipeline import Pipeline

# Create and run full pipeline
housing_prepared = full_pipeline.fit_transform(housing)
```

4. Train and evaluate models:
```python
# Train Random Forest with GridSearch
grid_search.fit(housing_prepared, housing_label)

# Get predictions
final_predictions = final_model.predict(X_test_prepared)
```

## Data Visualization
The project includes various visualizations:
- Geographic distribution of housing prices
- Feature correlation matrix
- Scatter plots of key features
- Income category distribution
- Feature importance plots

## Model Performance
Models are evaluated using RMSE scores. The RandomForest model with GridSearchCV optimization typically provides the best performance.

## Directory Structure
```
├── datasets/
│   └── housing/        # Housing dataset directory
├── notebooks/         # Jupyter notebooks
└── src/              # Source code
    ├── data/         # Data processing scripts
    └── models/       # Model implementation
```
