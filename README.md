# Titanic Survival Prediction

This repository contains a complete workflow for predicting Titanic passenger survival using machine learning. It includes data cleaning, feature engineering, model training, evaluation, hyperparameter tuning, and submission file generation for Kaggle.

## Directory Structure

```
Titanic/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── submission_log.csv
│   ├── submission_rf.csv
├── figures/
│   ├── titanic_survival_*.png
├── model/
│   ├── titanic_log_model.joblib
│   ├── titanic_rf_model.joblib
├── notebooks/
│   └── titanic_survival.ipynb
├── reports/
│   └── titanic_survival.pdf
└── README.md
```

## Main Files

- **notebooks/titanic_survival.ipynb**  
  Jupyter notebook with all code for data analysis, feature engineering, model building, evaluation, and submission generation.

- **data/train.csv, data/test.csv**  
  Raw Titanic datasets from Kaggle.

- **data/submission_log.csv, data/submission_rf.csv**  
  Submission files for Logistic Regression and Random Forest models.

- **model/titanic_log_model.joblib, model/titanic_rf_model.joblib**  
  Saved model pipelines for Logistic Regression and Random Forest (including preprocessing).

- **figures/**  
  Plots and visualizations generated during EDA and model evaluation.

- **reports/titanic_survival.pdf**  
  PDF report summarizing the workflow, results, and findings.

## Workflow Overview

1. **Data Loading & Exploration**
   - Loads train and test datasets.
   - Explores missing values and distributions.

2. **Feature Engineering**
   - Extracts titles from names.
   - Creates family size and "IsAlone" features.

3. **Data Cleaning**
   - Imputes missing values using domain knowledge.
   - Drops unnecessary columns.

4. **Preprocessing Pipeline**
   - Scales numeric features.
   - One-hot encodes categorical features.

5. **Model Training**
   - Trains Logistic Regression and Random Forest using pipelines.
   - Evaluates with accuracy, classification report, ROC AUC, and confusion matrix.

6. **Hyperparameter Tuning**
   - Uses GridSearchCV to tune Random Forest parameters.

7. **Model Selection**
   - Compares validation and Kaggle leaderboard scores.
   - Saves both models for reproducibility.

8. **Submission Generation**
   - Predicts on test set and generates submission files for both models.

9. **Saving Models**
   - Saves trained pipelines for future use.

## Results Summary

| Model                  | Validation Accuracy | ROC AUC  |
|------------------------|------------------|----------|
| Logistic Regression     | 85.5%             | 0.878    |
| Random Forest           | 83.2%             | 0.843    |

> The Logistic Regression model performed slightly better on validation data, but both models produced strong submissions for the Kaggle Titanic competition.


## How to Use

1. Clone the repository.
2. Place `train.csv` and `test.csv` in the `data/` directory.
3. Run the notebook `notebooks/titanic_survival.ipynb` step by step.
4. Find submission files in `data/` and trained models in `model/`.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- joblib
- Jupyter Notebook

Install dependencies with:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Credits

- Titanic dataset: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Author: Vishal

---