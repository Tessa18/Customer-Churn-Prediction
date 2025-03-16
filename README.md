# Customer-Churn-Prediction

## Project Overview
This project aims to predict customer churn using various machine learning models. The dataset includes customer-related features, and the goal is to identify customers who are likely to leave the service.

## Dataset
-  **Source**: Churn_Modelling.csv
-  **Target Variable**: Exited (1 = Churn, 0 = Retained)
-  **Preprocessing Steps**:
    * Removed irrelevant columns (RowNumber, CustomerId, Surname)
    * Created additional features like:
        * Balance to Salary Ratio (Balance / (EstimatedSalary + 1))
        * Tenure by Age (Tenure / (Age + 1))
    * Encoded categorical variables
    * Split the data into training, validation, and test sets (70%-15%-15%)

## Models Used
-  Logistic Regression
-  Decision Tree
-  XGBoost
-  LightGBM
-  Random Forest (with GridSearchCV tuning)
-  Hyperparameter tuning:
     * Used GridSearchCV for Random Forest.
     * Manually tuned XGBoost with validation set.
 
## Model Evaluation
Each model was trained and evaluated using:

-  Accuracy
-  ROC-AUC Score
-  Confusion Matrix
-  Classification Report

The best model was selected based on performance on the validation set and was then trained on the full dataset.

## Results
-  Model performance comparisons were made using validation set metrics.
-  The final model was trained using the best hyperparameter-tuned model.

## Usage
To run the project:

1. Install dependencies:
-      pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn

2.Load and preprocess the dataset.

3.Train models and evaluate performance.

## Future Improvements
-  Feature selection and engineering.
-  Try additional ensemble methods.
-  Deploy the best model as a web service.
