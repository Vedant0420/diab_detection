# Diabetes Outcome Classification

TYBCA Machine Learning Project (CCA2), Problem Statement 8.

Predicts whether a patient has diabetes from 8 health measurements, and compares
four classification algorithms: Logistic Regression, KNN, Decision Tree and Random Forest.

## Dataset

Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases).
768 patients (all female, Pima Indian heritage, age 21+), 8 features, target `Outcome` (1 = diabetes).

- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Copy used here: `diabetes.csv`

## Files

| File | What it is |
|---|---|
| `diabetes_prediction.ipynb` | Jupyter notebook with code, charts and outputs |
| `diabetes_prediction.py` | Same code as a plain Python script |
| `diabetes.csv` | Dataset |
| `outputs/` | Saved charts, model comparison table, confusion matrix, run output |
| `requirements.txt` | Python libraries needed |

## How to run

```
pip install -r requirements.txt
python diabetes_prediction.py
```

Or open `diabetes_prediction.ipynb` in Jupyter / VS Code and run all cells.

## Results (test set, 154 patients)

| Model | CV F1 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.672 | 0.734 | 0.603 | 0.704 | 0.650 |
| KNN (k=11) | 0.611 | 0.734 | 0.633 | 0.574 | 0.602 |
| Decision Tree | 0.637 | 0.708 | 0.556 | 0.833 | 0.667 |
| **Random Forest** | **0.697** | 0.721 | 0.580 | 0.741 | 0.650 |

Final model: **Random Forest** (best 5-fold cross-validated F1). Most important features: Glucose, BMI, Age.

## Limitations

- Small dataset (768 rows) from one population (Pima Indian women), so results may not generalise to men or other groups.
- Many missing values (Insulin 49%, SkinThickness 30%) were filled with the median.
- Not a medical diagnosis tool. A false negative (missing a diabetic patient) is the most harmful error.
