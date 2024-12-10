# Space Transport Prediction Project

## Overview
This project implements a machine learning pipeline to predict passenger transportation outcomes in space travel using XGBoost classification. It features custom transformers for data preprocessing and handles various types of input data including numerical, binary, and categorical features.

## Features
- Custom transformation pipeline
- Numerical data imputation
- Binary data conversion
- Categorical data encoding
- XGBoost classification
- Model performance evaluation using ROC AUC

## Dependencies
- pandas
- numpy
- xgboost
- scikit-learn

## Dataset
The project uses a space transport dataset with the following key features:
- PassengerId
- HomePlanet
- CryoSleep
- Cabin
- Destination
- Age
- VIP
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck
- Name
- Transported (target variable)

## Custom Transformers
The project implements three custom transformer classes:

1. `Numerical_Imputer`:
   - Handles missing values in numerical columns
   - Uses median strategy for imputation
   - Processes: Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck

2. `Binary_converter`:
   - Converts boolean values to binary (0/1)
   - Processes: CryoSleep

3. `OneHotEncode`:
   - Performs one-hot encoding for categorical variables
   - Processes: HomePlanet, Destination, VIP

## Pipeline Architecture
```python
pipeline = Pipeline(steps=[
    ('numerical_imputer', Numerical_Imputer()),
    ('binary_converter', Binary_converter()),
    ('one_hot_encoder', OneHotEncode())
])
```

## Model Configuration
XGBoost Classifier parameters:
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 6
- random_state: 42

## Usage

1. Install required packages:
```bash
pip install pandas numpy xgboost scikit-learn
```

2. Load and prepare the data:
```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X = train_data.drop(['Transported', 'PassengerId', 'Name', 'Cabin'], axis=1)
y = train_data['Transported']
```

3. Split the data:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

4. Run the pipeline and train the model:
```python
# Transform data
X_transformed = pipeline.fit_transform(X_train)

# Train model
xgb.fit(X_transformed, y_train)

# Make predictions
y_pred = xgb.predict(pipeline.transform(X_test))
```

5. Evaluate the model:
```python
y_pred_proba = xgb.predict_proba(pipeline.transform(X_test))[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
```

## Model Performance
The model's performance is evaluated using ROC AUC score, which measures the model's ability to distinguish between transportation outcomes.

## Data Preprocessing Steps
1. Removal of irrelevant columns (PassengerId, Name, Cabin)
2. Imputation of missing values in numerical columns
3. Conversion of boolean values to binary integers
4. One-hot encoding of categorical variables
5. Feature transformation through the custom pipeline
