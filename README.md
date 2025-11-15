# Income Prediction Using U.S. Census Data

## Project Overview
This project uses the 1994 U.S. Adult Census dataset to build a machine learning model that predicts whether an individual's annual income exceeds $50,000. The notebook includes exploratory data analysis (EDA), preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation using Logistic Regression and Random Forest classifiers.

The goal is to understand socioeconomic patterns associated with higher income levels and to create a predictive model that performs well on unseen data.

## Dataset
**Target Variable:** income_binary (<=50K or >50K)


### Numerical Features:
- age
- education-num
- capital-gain
- capital-loss
- hours-per-week

### Categorical Features:
- workclass
- marital-status
- occupation
- relationship
- race
- sex
- native-country

## Data Preprocessing
- Replaced missing “?” values with NaN
- Imputed missing values (median for numeric, mode for categorical)
- One-hot encoded categorical variables
- Standardized numerical columns
- Removed redundant or uninformative features (`fnlwgt`, duplicate `education`)
- Used scikit-learn ColumnTransformer + Pipeline for clean, reproducible preprocessing

## Models Implemented
### Logistic Regression
- Used as a baseline model
- Applied class_weight='balanced'
- Tuned regularization strength (C) using GridSearchCV

### Random Forest Classifier
- Captured non-linear relationships
- Tuned hyperparameters (n_estimators, max_depth)

## Model Evaluation
Metrics used for comparison:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve Plot
- Feature Importances (Random Forest)

## Key Insights
- education-num, capital-gain, hours-per-week, and age were strong predictors
- Both ML models achieved strong ROC-AUC performance
- Pipelines ensured consistent preprocessing across train/test splits

  ## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook






