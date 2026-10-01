
# 🏦 Credit Risk Application Scorecard

An end-to-end credit risk scorecard pipeline — from raw applicant data to a 
deployed, explainable prediction app — built to replicate real-world bureau 
scorecard methodology (WOE/IV, logistic regression, reason codes).

🔗 **Live App:** [link to your deployed Streamlit app]
📓 **Notebook:** `02_notebooks/model.ipynb`

---

## ⚠️ Important Disclaimer

This model is trained on a **public, simulated dataset** (Kaggle: *Credit Risk 
Dataset* by Lao Tse, CC0 license) that simulates credit bureau-style data. 
Real bank bureau data is not publicly available due to privacy regulation, so 
this project demonstrates **methodology**, not a production-ready credit model. 
It should not be used for actual lending decisions.

---

## 📌 Project Overview

This project builds a credit risk **application scorecard** — the type of 
model banks use to decide whether to approve, reject, or manually review a 
loan application — using:

- Weight of Evidence (WOE) encoding + Information Value (IV) feature selection
- Logistic regression (industry standard for scorecards, due to interpretability 
  and regulatory explainability requirements)
- A points-based credit score (Offset/Factor/PDO scaling, FICO-style)
- Reason codes for every prediction (ECOA-style adverse action explanations)
- An interactive Streamlit app for live prediction + model performance dashboard

---

## 📊 Dataset

**Source:** [Credit Risk Dataset by Lao Tse (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
**License:** CC0 Public Domain
**Size:** ~32,500 rows, 11 features + 1 target (`default_flag`)

| Category | Variables |
|---|---|
| Demographics | age, home_ownership, employment_years |
| Financial | annual_income |
| Loan Information | loan_purpose, loan_amount, interest_rate |
| Credit Bureau | credit_grade, credit_history_years, prior_default_flag |
| Engineered | loan_income_ratio (loan_amount / annual_income) |
| Target | default_flag (0 = repaid, 1 = defaulted) |

---

## 🔍 Methodology

### 1. Data Cleaning
- Removed 165 exact duplicate rows (no ID column, full-row match)
- Removed logically impossible values (`age ≥ 100`, `employment_years ≥ age`)
- These steps were performed **before** the train/test split, since they're 
  rule-based corrections, not data-derived statistics

### 2. Train-Test Split
- 80/20 split, **stratified** on `default_flag` (target imbalance: 78.1% / 21.9%)
- All subsequent steps (imputation, binning, WOE) were fit on the training 
  set only, to prevent data leakage

### 3. Missing Value Handling
- `employment_years` (2.75% missing), `interest_rate` (9.55% missing)
- Median imputation (fit on train only)

### 4. Binning
- **Quantile (equal-frequency) binning** chosen over decision-tree binning, 
  after tree-based bins produced unstable zero-count bins for the WOE 
  calculation on this dataset's class distribution
- Verified **monotonicity** of bad rate across bins post-binning — a required 
  property for stable WOE and reliable logistic regression coefficients

### 5. WOE Encoding & Feature Selection
- Weight of Evidence computed per bin/category (fit on train + y_train only)
- Information Value used to rank feature strength; features with IV > 0.5 
  were specifically investigated for leakage/redundancy (see below)

### 6. ⚠️ Leakage & Redundancy Investigation
- `credit_grade` and `interest_rate` both showed suspiciously high IV 
  (>0.5). Investigation confirmed `interest_rate` is largely derived from 
  `credit_grade` (grade-based pricing), creating redundant signal.
- Two model variants were built and compared: **Full model** (includes 
  `credit_grade`) vs. **Reduced model** (excludes it), to separately evaluate 
  how much predictive power comes from the pre-existing bureau grade vs. 
  features engineered from first principles.

### 7. Class Imbalance
- Target is imbalanced (21.9% default rate) — realistic for unsecured retail 
  lending
- Logistic regression coefficients are relatively robust to imbalance, but 
  the **classification threshold** is not — addressed via precision-recall 
  curve analysis and F2-score optimization (recall-weighted, reflecting that 
  missed defaulters are costlier than wrongly-rejected good applicants)

### 8. Model Evaluation
| Metric | Value |
|---|---|
| AUC | 0.87 |
| Gini | 0.75 |
| KS Statistic | 0.22 |
| Threshold (F2-optimal) | 0.18 |

### 9. Scorecard Conversion
- PD converted to a points-based score: `Score = Offset + Factor × ln(odds)`
- Offset = 600, PDO (points to double odds) = 20

---

## 🖥️ Streamlit App

**Tab 1 — Predict & Explain**
- Input form grouped by Demographic / Financial / Credit Bureau sections
- Outputs: Probability of Default, Credit Score, Risk Tier, Approve/Review/Reject decision
- Reason codes: WOE × coefficient contribution per feature, explaining *why* 
  a decision was made (ECOA-style explainability)

**Tab 2 — Model Dashboard**
- IV ranking, ROC/PR curves, KS chart, confusion matrix, calibration curve

---

## 🛠️ Tech Stack
- Python, pandas, scikit-learn, feature-engine (WOE/binning)
- Streamlit + Plotly (app + interactive charts)
- joblib/pickle (model persistence)

---

## 📂 Project Structure
```
credit-risk-scorecard/
├── 01_data/
│   └── credit_risk_clean.csv
├── 02_notebooks/
│   └── model.ipynb
├── 03_models/
│   ├── median_imputer.pkl
│   ├── binner.pkl
│   ├── woe_encoder.pkl
│   ├── logreg_model.pkl
│   └── config.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚧 Limitations & What Would Change in Production

- **Synthetic data:** results reflect dataset simplicity, not real-world noise
- **No reject inference:** real scorecards must account for declined 
  applicants whose outcome is never observed
- **No population stability monitoring (PSI):** real scorecards are monitored 
  over time for score-distribution drift; not applicable without time-series data
- **Interest rate assumption:** in the live app, interest rate is derived from 
  credit grade via an assumed pricing table, since a real applicant wouldn't 
  know their rate before underwriting
- **Protected attributes:** `age` is included for demonstration; in production, 
  protected demographic attributes are governed by fair-lending regulation 
  (ECOA in the US, RBI Fair Practices Code in India)

---

## 👤 Author
Manish Deswal 
Data Science Student, Delhi University | Aspiring Credit Data Scientist
[LinkedIn] | [GitHub] | [Portfolio]
```

---

