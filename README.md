# Bank GoodCredit — Credit Risk Prediction

A machine learning project that predicts whether a bank customer has a **good or bad credit history**, to help a lender identify higher-risk customers.

> ⚠️ **Data note:** The client's MySQL database was not accessible during development (connection error 1045). The notebook therefore runs on a **synthetic demo dataset** that mimics the original schema. All scores, feature importances and decile results are **illustrative only** and must not be used for real lending decisions.

## Business Objective

Predict the probability that a customer has a bad credit history.

- **Target:** `Bad_label`
  - `0` — Good credit history
  - `1` — Bad credit history (30+ days past due)

## Data

Three tables, linked by `customer_no` (original source: MySQL database `project_banking`; demo equivalents are generated in the notebook):

| Table | Level | Description |
|---|---|---|
| `cust_demographics` | Customer | ID, entry date, target, 79 anonymized features |
| `cust_account` | Account | Account type, credit limit, balance, past-due and payment amounts |
| `cust_enquiry` | Enquiry | Enquiry date, purpose, amount |

Demo dataset size: 1,500 customers · 3,714 account records · 3,079 enquiry records · ~22.5% bad-credit rate.

## Workflow

1. Database connection (MySQL via `python-dotenv`) and fallback to demo data
2. Data understanding and EDA (types, missing values, duplicates, target distribution)
3. Feature engineering
   - `credit_utilization` = current balance / credit limit
   - `has_past_due` flag
   - Customer-level aggregates: number of accounts, average utilization, total balance, total past due, accounts with past due, enquiry count / average / total amount
4. Master dataset: left-join of all tables on `customer_no` (1,500 rows × 90 columns)
5. Train/test split (80/20, stratified) and standardization
6. Model training: Logistic Regression, Decision Tree, Random Forest
7. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Gini, confusion matrix, feature importance, decile rank ordering

## Results (synthetic demo data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini |
|---|---|---|---|---|---|---|
| Random Forest | 0.763 | 0.333 | 0.044 | 0.078 | 0.707 | 0.414 |
| Logistic Regression | 0.767 | 0.462 | 0.176 | 0.255 | 0.664 | 0.328 |
| Decision Tree | 0.663 | 0.351 | 0.574 | 0.436 | 0.662 | 0.324 |

**Key observations**

- Random Forest ranks customers best (highest ROC-AUC / Gini), but its **recall for bad-credit customers is very low** at the default 0.5 threshold — it misses most risky customers. Threshold tuning or cost-sensitive evaluation is needed.
- Decision Tree catches the most bad-credit customers (highest recall) at the cost of accuracy.
- Top Random Forest features were engineered ones: `total_past_due`, `accounts_with_past_due`, `avg_credit_utilization`, `total_current_balance`.
- Decile ranking: the highest-risk decile had a much higher observed bad rate (50%) than the lowest (6.7%), showing the model rank-orders risk reasonably even though classification at 0.5 is weak.

## Limitations & Next Steps

- Results come from synthetic data — validate on real, authorized data
- Tune hyperparameters and the decision threshold; try gradient boosting (XGBoost / LightGBM)
- Test on out-of-time data, check probability calibration
- Review data leakage, fairness and explainability (e.g. SHAP)
- Add monitoring and approval documentation before any deployment

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · MySQL Connector · Jupyter

## How to Run

```bash
git clone https://github.com/<your-username>/bank-goodcredit-credit-risk.git
cd bank-goodcredit-credit-risk

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

jupyter notebook Bank_GoodCredit_Credit_Risk_Prediction.ipynb
```

The notebook generates its own demo data, so **no database credentials are needed** to reproduce the results. To connect to a real MySQL database, create a local `.env` file (never commit it):

```
DB_HOST=your_host
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=project_banking
```

## Project Structure

```
bank-goodcredit-credit-risk/
├── Bank_GoodCredit_Credit_Risk_Prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Trisha** — AI Engineer Trainee, DataMites
B.Tech CSE · Python · SQL · Machine Learning · Data Visualization

[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
