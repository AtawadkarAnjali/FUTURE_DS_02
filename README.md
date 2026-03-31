# FUTURE_DS_02

<img width="1203" height="688" alt="Screenshot 2026-03-22 162358" src="https://github.com/user-attachments/assets/0a49b1dc-6a83-4e12-8b56-cdae9afa3164" />
<img width="464" height="461" alt="Screenshot 2026-03-22 162317" src="https://github.com/user-attachments/assets/2a6584d7-3a39-42bd-939a-996be5e91e30" />
<img width="686" height="653" alt="Screenshot 2026-03-22 162418" src="https://github.com/user-attachments/assets/6e3fd70c-3c36-4f9c-8cb0-8a4f19f39154" />
<img width="542" height="424" alt="Screenshot 2026-03-22 162336" src="https://github.com/user-attachments/assets/b0081940-a873-4bee-bf36-46176a85e23f" />


# 📞 Customer Churn Prediction — Telecom (California, Q2 2022)

> Predicting whether a telecom customer will **Churn**, **Stay**, or **Join** using machine learning on 7,043 real customer records.

---

## 📌 Project Overview

This project builds an end-to-end churn prediction pipeline for a California-based telecommunications company. The dataset contains demographic, service subscription, billing, and behavioral data for Q2 2022 customers.

**Target Variable:** `Customer Status` — 3 classes: `Churned`, `Stayed`, `Joined`

---

## 📂 Dataset

| Attribute | Detail |
|---|---|
| Total Records | 7,043 customers |
| Original Features | 38 columns |
| After Preprocessing | 31 features (7 dropped) |
| Final Feature Matrix | 1,129 columns (after one-hot encoding) |
| Usable Rows (after null handling) | 4,835 |

---

## ⚙️ Pipeline

### 1. Data Preprocessing
- Dropped 7 leakage/identifier columns (`Customer ID`, `Churn Category`, `Churn Reason`, etc.)
- Missing values: ~9.7% in phone-related features, ~21.7% in internet-related features
- Treatment: `interpolate()` followed by `dropna()`

### 2. EDA
- Age distribution by customer status (histogram)
- Correlation heatmap (Pearson)
- Boxplots of 11 numeric features vs Customer Status
- Demographic cross-tabs (Gender, Married)
- Offer & Payment Method breakdowns

### 3. Feature Engineering
- **Label Encoding:** Binary columns (Gender, Married, Yes/No service features)
- **One-Hot Encoding:** Payment Method, Contract, Internet Type, Offer, City (1,106 cities)
- **MinMaxScaler:** Applied to 11 continuous numeric columns
- **Train/Test Split:** 80/20 (3,868 train / 967 test), `random_state=5`

### 4. Model Benchmarking (GridSearchCV + ShuffleSplit CV)

| Model | Best CV Score | Best Param |
|---|---|---|
| **XGBoost Classifier** | **82.73%** | base_score=0.5 |
| Logistic Regression | 78.28% | C=5 |
| Random Forest | 78.12% | n_estimators=10 |
| Decision Tree | 77.29% | criterion=gini |
| Gaussian Naive Bayes | 36.77% | — |

---

## 🏆 Final Model: XGBoost Classifier

**Test Accuracy: 80.87%**

| Class | Label | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| 0 | Churned | 0.77 | 0.66 | 0.71 | 348 |
| 1 | Joined | 0.81 | 0.44 | 0.57 | 50 |
| 2 | Stayed | 0.82 | 0.93 | 0.87 | 569 |
| — | Weighted Avg | 0.81 | 0.81 | 0.80 | 967 |

---

## 💡 Key Business Insights

- **Tenure:** Short-tenure customers churn most — prioritise retention in months 1–12
- **Contract Type:** Month-to-Month = highest churn risk; incentivise annual upgrades
- **Monthly Charge:** Higher charges correlate with churn — targeted discount programs help
- **Promotions:** Customers with no offer have the highest churn exposure
- **Referrals:** High referral count = strong stay signal; referral programs boost retention

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn` `Plotly`

---

## 🚀 Future Improvements

- Apply SMOTE to address class imbalance for the `Joined` class (Recall: 0.44)
- Use target encoding or geographic clustering instead of 1,106 city dummies
- Tune XGBoost hyperparameters (n_estimators, max_depth, learning_rate)
- Use StratifiedKFold instead of ShuffleSplit for class-balanced CV
