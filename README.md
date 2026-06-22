# Adult Census Income Prediction

A binary classification project that predicts whether a person earns **>50K or ≤50K per year** based on demographic and employment data from the 1994 US Census.

Three ML models are compared through a single end-to-end sklearn pipeline — Logistic Regression, Gradient Boosting, and Random Forest — with Random Forest coming out on top at **85.65% test accuracy and ROC-AUC of 0.9097**.

---

## Dataset

**Source:** [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult) (Kohavi & Becker, 1994 US Census Bureau)

| Split | Rows |
|-------|------|
| Training | 32,561 |
| Test | 16,281 |

**Features (14):** age, workclass, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

**Target:** `income` → `>50K` (1) or `≤50K` (0)

---

## Project Workflow

```
Step 1 → Import Libraries
Step 2 → Load Dataset
Step 3 → Exploratory Data Analysis (EDA)
Step 4 → Data Preprocessing
Step 5 → Build ML Pipeline
Step 6 → Train the Model
Step 7 → Evaluate (Metrics + Confusion Matrix + ROC Curve + Feature Importance)
Step 8 → Save & Reload Model (joblib)
Step 9 → Predict on New / Batch Data
```

---

## Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, joblib

---

## Pipeline Architecture

The entire preprocessing and modeling logic is wrapped in a single `sklearn.pipeline.Pipeline`.

```
ColumnTransformer
├── Numerical: SimpleImputer (median) → StandardScaler
└── Categorical: SimpleImputer (most_frequent) → OneHotEncoder

Pipeline
├── preprocessor: ColumnTransformer (above)
└── classifier: RandomForestClassifier / GradientBoostingClassifier / LogisticRegression
```

---

## Results

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 84.10% | 0.88 |
| Gradient Boosting | 85.02% | 0.90 |
| **Random Forest** | **85.65%** | **0.9097** |

**Random Forest config:** `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `random_state=42`

**Cross-validation:** 5-fold stratified CV on training set

**Top predictors (by feature importance):**
1. marital-status (married-civ-spouse)
2. capital-gain
3. education level

---

## Files

```
├── Predict_Adult_Income.ipynb           # Main notebook (full pipeline)
├── adult.data                           # Training set
├── adult.test                           # Test set
├── adult_income_pipeline.pkl            # Saved model (joblib)
├── Adult_Income_ML_Project_Report.pdf   # Full project report
└── Comparative_Performance_Analysis...pdf  # Research paper format
```

---

## How to Run

1. Open `Predict_Adult_Income.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `adult.data` and `adult.test` when prompted
3. Run all cells top to bottom
4. The trained pipeline gets saved as `adult_income_pipeline.pkl` and downloaded automatically

---

## Sample Prediction

```python
import joblib, pandas as pd

pipeline = joblib.load("adult_income_pipeline.pkl")

new_person = pd.DataFrame([{
    "age": 35, "workclass": "Private", "education": "Bachelors",
    "education_num": 13, "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial", "relationship": "Husband",
    "race": "White", "sex": "Male", "capital_gain": 5000,
    "capital_loss": 0, "hours_per_week": 45, "native_country": "United-States"
}])

prediction = pipeline.predict(new_person)
probability = pipeline.predict_proba(new_person)
print(">50K" if prediction[0] == 1 else "<=50K")
```

---

## Key Takeaways

- Ensemble methods (Random Forest, Gradient Boosting) outperform linear classifiers on this structured tabular dataset
- A clean sklearn pipeline makes preprocessing fully reproducible and deployment-ready
- Marital status, capital gain, and education are the strongest income predictors in this dataset

---

## References

- Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. KDD.
- UCI ML Repository: [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics.
