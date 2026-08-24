# Fraud Detection System

An end-to-end Machine Learning system for detecting fraudulent financial transactions using the PaySim dataset.

The project covers the complete ML lifecycle: data analysis, preprocessing, feature engineering, model training and comparison, class imbalance analysis, threshold optimization, explainability with SHAP, robustness validation, production API development, an interactive dashboard, automated testing, Docker containerization, and cloud deployment.

---

## Production Dashboard

The deployed Streamlit dashboard provides an interactive interface for analyzing financial transactions in real time.

Users can enter transaction information manually and receive:

- Fraud or legitimate classification
- Fraud probability score
- Transaction risk level
- Immediate fraud alerts

The example below shows a suspicious `TRANSFER` transaction detected by the production model with a **99.69% fraud probability** and a **CRITICAL** risk level.

![Fraud Detection Dashboard](images/dashboard_demo.png)

---

## Live Demo

### Streamlit Dashboard

https://cozy-communication-production-435b.up.railway.app

### FastAPI Swagger Documentation

https://fraud-detection-system-production-b261.up.railway.app/docs

### API Health Check

https://fraud-detection-system-production-b261.up.railway.app/health

---

## Project Overview

Financial fraud detection is a highly imbalanced classification problem. In the PaySim dataset, fraudulent transactions represent only a very small percentage of all transactions.

The objective of this project was not only to train a machine learning model, but to build a complete production-style fraud detection system.

The system can:

- Analyze financial transaction data
- Detect fraudulent transactions
- Generate fraud probabilities
- Assign transaction risk levels
- Analyze individual transactions
- Analyze CSV batches of transactions
- Explain model behavior using SHAP
- Evaluate temporal generalization and potential data leakage
- Perform feature ablation analysis
- Serve predictions through a REST API
- Provide an interactive web dashboard
- Run through Docker containers
- Operate as a deployed cloud application

---

## System Architecture

```text
                     User
                      |
                      v
              Streamlit Dashboard
                      |
                      v
                 FastAPI API
                      |
                      v
             Input Validation
                      |
                      v
              Feature Engineering
                      |
                      v
          Production ML Pipeline
                      |
                      v
               Random Forest
                      |
                      v
          Fraud Probability Score
                      |
                      v
        Fraud / Legitimate Decision
                      |
                      v
                Risk Level
```

### Production Deployment

```text
                    Internet
                       |
                       v
             Railway Public Domain
                       |
                       v
              Streamlit Dashboard
                    :8501
                       |
                       |
             Railway Private Network
                       |
                       v
                 FastAPI API
                    :8000
                       |
                       v
          Fraud Detection Pipeline
```

---

## Dataset

The project uses the **PaySim synthetic financial transaction dataset**.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Transactions | 6,362,620 |
| Fraud Transactions | 8,213 |
| Legitimate Transactions | 6,354,407 |
| Fraud Percentage | 0.1291% |
| Original Features | 11 |

The extreme imbalance makes accuracy alone unsuitable for evaluating the fraud detection system.

For this reason, the project focuses primarily on:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- False Positives
- False Negatives

### Transaction Types

The dataset contains:

- CASH_OUT
- PAYMENT
- CASH_IN
- TRANSFER
- DEBIT

Fraudulent transactions occur in:

- TRANSFER
- CASH_OUT

The full PaySim dataset is not stored in this repository because of its size.

---

## Exploratory Data Analysis

EDA was performed to understand transaction behavior and fraud patterns.

Important observations included:

- Fraud represents only 0.1291% of all transactions.
- Fraud appears only in TRANSFER and CASH_OUT transactions.
- Fraudulent transactions generally involve larger transaction amounts.
- Sender and receiver balance behavior provides strong signals for fraud detection.

### Transaction Type Distribution

![Transaction Type Distribution](images/transaction_type_distribution.png)

### Fraud by Transaction Type

![Fraud by Transaction Type](images/fraud_by_transaction_type.png)

### Fraud Rate by Transaction Type

![Fraud Rate by Transaction Type](images/fraud_rate_by_type.png)

---

## Data Splitting

The primary model-development workflow used a stratified train/validation/test split.

| Dataset | Transactions | Fraud Cases |
|---|---:|---:|
| Training | 4,453,834 | 5,749 |
| Validation | 954,393 | 1,232 |
| Test | 954,393 | 1,232 |

The final test set remained untouched until the final production evaluation.

A separate temporal holdout experiment was later performed to evaluate generalization from earlier transactions to future transactions.

---

## Feature Engineering

Several fraud-specific features were created from the original transaction information.

### Engineered Features

- `orig_balance_change`
- `dest_balance_change`
- `orig_balance_error`
- `dest_balance_error`
- `amount_to_orig_balance`
- `orig_account_emptied`

These features capture behavioral patterns in transaction amounts and account balance changes.

Feature engineering produced a major improvement in fraud detection performance.

### Raw vs Engineered Features

| Feature Set | Precision | Recall | F1 Score | PR-AUC | Missed Fraud |
|---|---:|---:|---:|---:|---:|
| Raw Features | 0.9792 | 0.7646 | 0.8587 | 0.9375 | 290 |
| Engineered Features | 1.0000 | 0.9968 | 0.9984 | 0.9979 | 4 |

The comparison demonstrates that the engineered features substantially improved the model's ability to detect fraudulent transactions.

---

## Machine Learning Models

Three main machine learning models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Logistic Regression

Logistic Regression was used as the baseline model.

| Metric | Score |
|---|---:|
| Precision | 0.9286 |
| Recall | 0.5698 |
| F1 Score | 0.7062 |
| ROC-AUC | 0.9962 |
| PR-AUC | 0.7975 |
| Missed Fraud | 530 |

Although ROC-AUC was high, recall was insufficient for fraud detection.

---

## Random Forest

Random Forest produced a major improvement over the baseline.

| Metric | Score |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.9968 |
| F1 Score | 0.9984 |
| ROC-AUC | 0.9996 |
| PR-AUC | 0.9979 |
| Missed Fraud | 4 |

Validation confusion matrix:

```text
True Positives:     1,228
False Positives:        0
False Negatives:        4
True Negatives:   953,161
```

### Random Forest Confusion Matrix

![Random Forest Confusion Matrix](images/random_forest_confusion_matrix.png)

---

## XGBoost

XGBoost was also trained and evaluated.

| Metric | Score |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.9959 |
| F1 Score | 0.9980 |
| ROC-AUC | 0.9997 |
| PR-AUC | 0.9983 |
| Missed Fraud | 5 |

### XGBoost Confusion Matrix

![XGBoost Confusion Matrix](images/xgboost_confusion_matrix.png)

---

## Model Comparison

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Missed Fraud |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9286 | 0.5698 | 0.7062 | 0.9962 | 0.7975 | 530 |
| Random Forest | 1.0000 | 0.9968 | 0.9984 | 0.9996 | 0.9979 | 4 |
| XGBoost | 1.0000 | 0.9959 | 0.9980 | 0.9997 | 0.9983 | 5 |

Random Forest achieved the highest validation F1 score and the lowest number of missed fraudulent transactions.

---

## Handling Class Imbalance

The dataset contains approximately:

```text
Normal Transactions : Fraud Transactions
773.71 : 1
```

XGBoost experiments were performed using different `scale_pos_weight` values:

- 1
- 10
- 50
- 100
- 300
- 773.71

The experiments showed that aggressive class weighting did not improve the overall F1 score compared with the already strong unweighted model.

This demonstrates why imbalance strategies should be evaluated experimentally rather than applied automatically.

---

## Threshold Optimization

The classification threshold was analyzed instead of relying only on the default threshold of `0.50`.

### Random Forest

Best validation F1 result:

```text
Threshold: 0.24
Precision: 1.0000
Recall:    0.9968
F1 Score:  0.9984
FP:        0
FN:        4
```

### XGBoost

A lower threshold increased recall while introducing a small number of false positives:

```text
Threshold: 0.06
Precision: 0.9976
Recall:    0.9968
F1 Score:  0.9972
FP:        3
FN:        4
```

This demonstrates the trade-off between detecting more fraud and generating false alarms.

### Random Forest Threshold Optimization

![Random Forest Threshold Optimization](images/random_forest_threshold_optimization.png)

### XGBoost Threshold Optimization

![XGBoost Threshold Optimization](images/xgboost_threshold_optimization.png)

---

## Model Evaluation

Because fraud is extremely rare, Precision-Recall analysis is particularly important.

### Precision-Recall Curve

![Precision Recall Curve](images/precision_recall_curve.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

---

## Explainable AI with SHAP

SHAP was used to understand which features influence fraud predictions.

### Top SHAP Features

The most influential features included:

1. `orig_balance_error`
2. `amount_to_orig_balance`
3. `orig_account_emptied`
4. `newbalanceOrig`
5. `orig_balance_change`

For one example fraudulent transaction, the model produced:

```text
Fraud Probability: 99.82%
```

The SHAP explanation showed which transaction characteristics increased the fraud prediction.

### SHAP Feature Importance

![SHAP Feature Importance](images/shap_feature_importance.png)

### SHAP Summary

![SHAP Summary](images/shap_summary.png)

SHAP makes the system more interpretable by providing insight into why the model considers transactions suspicious.

---

## Production Model

The production pipeline combines:

```text
Raw Transaction
      |
      v
Feature Engineering
      |
      v
Categorical Encoding
      |
      v
Random Forest
      |
      v
Fraud Probability
```

The complete pipeline is stored as:

```text
models/fraud_detection_pipeline.joblib
```

This allows preprocessing and inference to be performed consistently in production.

---

## Final Holdout Test

The production pipeline was evaluated on the untouched test dataset containing:

```text
954,393 transactions
1,232 fraud cases
```

### Final Results

| Metric | Result |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.9959 |
| F1 Score | 0.9980 |
| ROC-AUC | 0.9987 |
| PR-AUC | 0.9960 |

### Final Confusion Matrix

```text
True Positives:     1,227
False Positives:        0
False Negatives:        5
True Negatives:   953,161
```

The production model detected **1,227 of 1,232 fraudulent transactions** while generating **zero false positives** on the final holdout test at the evaluated threshold.

Because these results are unusually high, additional robustness and leakage analyses were performed below rather than assuming that the benchmark performance directly represents real-world generalization.

---

## FastAPI Backend

The machine learning pipeline is served through a FastAPI REST API.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | API and model health |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Analyze one transaction |
| POST | `/predict-batch` | Analyze multiple transactions |

### Live API Documentation

https://fraud-detection-system-production-b261.up.railway.app/docs

---

## Example API Request

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181,
  "oldbalanceOrg": 181,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 0
}
```

### Example Response

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.9969,
  "threshold": 0.5,
  "risk_level": "CRITICAL"
}
```

---

## Streamlit Dashboard

The Streamlit interface provides two analysis modes.

### Single Transaction Analysis

Users can manually enter transaction information and receive:

- Fraud / Legitimate prediction
- Fraud probability
- Risk level

Example production prediction:

```text
Prediction:        FRAUD
Fraud Probability: 99.69%
Risk Level:        CRITICAL
```

### Batch CSV Analysis

Users can upload a CSV containing multiple transactions.

The dashboard provides:

- Total transactions
- Fraud detected
- Legitimate transactions
- Critical-risk transactions
- High-risk transactions
- Medium-risk transactions
- Low-risk transactions
- Transaction-level probabilities
- Detected fraud table
- Downloadable analysis results

A demonstration dataset is included:

```text
data/portfolio_demo.csv
```

---

## Automated Testing

The project contains automated tests covering the API, model, and preprocessing pipeline.

Tests include:

- Root endpoint
- Health endpoint
- Model information endpoint
- Fraud prediction
- Legitimate prediction
- Batch prediction
- Invalid transaction type handling
- Negative amount rejection
- Empty batch rejection
- Production model existence
- Model loading
- Probability prediction
- Feature engineering columns
- Fraud feature calculations
- Zero sender balance handling

Run the test suite:

```bash
python -m pytest tests -v
```

Result:

```text
15 passed
```

---

## Docker

The project is containerized using two Docker services.

### API Container

```text
Dockerfile.api
```

Runs:

```text
FastAPI + Uvicorn
Port 8000
```

### Dashboard Container

```text
Dockerfile.dashboard
```

Runs:

```text
Streamlit
Port 8501
```

### Docker Compose

The complete application can be started locally with:

```bash
docker compose build
docker compose up
```

Then open:

```text
Dashboard:
http://localhost:8501

API Documentation:
http://localhost:8000/docs

API Health:
http://localhost:8000/health
```

---

## Cloud Deployment

The complete system is deployed using Railway.

Two separate services are deployed:

```text
Streamlit Dashboard
        |
        | Railway Private Network
        v
FastAPI Service
        |
        v
Fraud Detection Model
```

The dashboard communicates with FastAPI internally through Railway private networking.

### Live Dashboard

https://cozy-communication-production-435b.up.railway.app

### Live API

https://fraud-detection-system-production-b261.up.railway.app

---

## Robustness and Leakage Analysis

Because the model achieved unusually high performance on the synthetic PaySim dataset, additional experiments were performed to investigate:

- Traditional overfitting
- Random-split optimism
- Temporal generalization
- Post-transaction information leakage
- Dominant feature shortcuts
- Synthetic dataset artifacts

### 1. Temporal Holdout Validation

The original workflow used a stratified random split. To test whether this produced overly optimistic results, an additional chronological evaluation was performed.

PaySim contains 743 simulation steps.

The temporal experiment used:

```text
Training: step <= 378
Testing:  step > 378
```

Temporal split statistics:

| Dataset | Transactions | Fraud Cases | Fraud Rate |
|---|---:|---:|---:|
| Temporal Train | 5,444,003 | 4,207 | 0.0773% |
| Temporal Test | 918,617 | 4,006 | 0.4361% |

Results:

| Evaluation | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Holdout | 1.0000 | 0.9959 | 0.9980 | 0.9987 | 0.9960 | 0 | 5 |
| Temporal Holdout | 1.0000 | 0.9988 | 0.9994 | 1.0000 | 1.0000 | 0 | 5 |

The model remained highly effective when trained on earlier transactions and evaluated exclusively on future transactions.

This indicates that the near-perfect performance is not explained by the random split alone.

### 2. Pre-Transaction Validation

The production-style feature set includes post-transaction balances such as:

- `newbalanceOrig`
- `newbalanceDest`

and engineered features derived from those values.

To determine whether these fields were responsible for the high performance, a separate pre-transaction experiment removed post-transaction information.

The model was restricted to:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `oldbalanceDest`
- `amount_to_orig_balance`

The same temporal split was used.

Results:

| Metric | Result |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.9998 |
| F1 Score | 0.9999 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| True Positives | 4,005 |
| False Positives | 0 |
| False Negatives | 1 |
| True Negatives | 914,611 |

Removing post-transaction information did not reduce performance.

This indicates that post-transaction balance information is not the primary reason for the near-perfect benchmark results.

### 3. Feature Ablation Study

An ablation study was performed to determine which features were responsible for the unusually high performance.

| Experiment | Precision | Recall | F1 Score | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Type only | 0.0000 | 0.0000 | 0.0000 | 0.0176 | 0 | 4,006 |
| Type + Amount | 0.9892 | 0.1378 | 0.2419 | 0.2419 | 6 | 3,454 |
| + Sender Old Balance | 0.9952 | 0.4658 | 0.6346 | 0.9033 | 9 | 2,140 |
| + Receiver Old Balance | 0.9785 | 0.6932 | 0.8115 | 0.8854 | 61 | 1,229 |
| + Amount-to-Balance Ratio | 1.0000 | 0.9998 | 0.9999 | ~1.0000 | 0 | 1 |

The largest performance increase occurred after introducing:

```text
amount_to_orig_balance = amount / oldbalanceOrg
```

Without this ratio, the model achieved approximately:

```text
F1 Score: 81.15%
Recall:   69.32%
```

After adding the ratio:

```text
F1 Score: 99.99%
Recall:   99.98%
```

An additional experiment using only:

```text
amount
oldbalanceOrg
amount_to_orig_balance
```

also achieved approximately:

```text
Precision: 100%
Recall:    99.98%
F1 Score:  99.99%
PR-AUC:    99.98%
```

This confirmed that the ratio itself contains an exceptionally strong fraud signal in PaySim.

### 4. Dataset Shortcut Analysis

A direct statistical analysis was performed to understand why `amount_to_orig_balance` is so predictive.

For fraudulent transactions:

```text
Median ratio: 1.0
25th percentile: 1.0
75th percentile: 1.0
```

Most importantly:

```text
Fraud transactions with ratio between 0.99 and 1.01:
97.63%

Legitimate transactions with ratio between 0.99 and 1.01:
0.15%
```

Therefore, approximately **97.6% of fraudulent transactions** in PaySim transfer an amount almost equal to the sender's entire original balance, while this behavior occurs in only approximately **0.15% of legitimate transactions**.

This creates an unusually strong separation between the two classes.

### Interpretation

The additional experiments found no clear evidence that the near-perfect benchmark performance is caused by traditional overfitting or direct target leakage.

The evidence showed that:

- Performance remained strong on a future temporal holdout.
- Random splitting was not responsible for the high performance.
- Removing post-transaction balance information did not reduce performance.
- Adding the `step` feature did not explain the performance.
- Feature ablation identified `amount_to_orig_balance` as the dominant predictive signal.
- Direct statistical analysis confirmed that this ratio follows an unusually strong fraud-specific pattern in PaySim.

The near-perfect results are therefore largely explained by a **dataset-specific shortcut in the synthetic PaySim benchmark**.

Because PaySim is synthetic, these results should **not** be interpreted as evidence that the model would achieve the same performance on real-world banking transactions.

This analysis demonstrates an important ML engineering principle:

> Extremely high benchmark performance should be investigated for leakage, temporal effects, unrealistic feature relationships, and dataset-specific shortcuts before being interpreted as real-world generalization.

---

## Project Structure

```text
fraud-detection-system/
|
|-- api/
|   |-- __init__.py
|   |-- main.py
|   `-- schemas.py
|
|-- dashboard/
|   `-- app.py
|
|-- data/
|   `-- portfolio_demo.csv
|
|-- images/
|   |-- dashboard_demo.png
|   |-- fraud_by_transaction_type.png
|   |-- fraud_rate_by_type.png
|   |-- precision_recall_curve.png
|   |-- random_forest_confusion_matrix.png
|   |-- random_forest_threshold_optimization.png
|   |-- roc_curve.png
|   |-- shap_feature_importance.png
|   |-- shap_summary.png
|   |-- transaction_type_distribution.png
|   |-- xgboost_confusion_matrix.png
|   `-- xgboost_threshold_optimization.png
|
|-- models/
|   |-- fraud_detection_pipeline.joblib
|   |-- metadata.json
|   |-- model_comparison.csv
|   |-- shap_feature_importance.csv
|   |-- threshold_results.csv
|   `-- ablation_study_results.csv
|
|-- src/
|   |-- inspect_data.py
|   |-- eda.py
|   |-- preprocessing.py
|   |-- features.py
|   |-- train.py
|   |-- train_random_forest.py
|   |-- train_xgboost.py
|   |-- xgboost_imbalance.py
|   |-- evaluate.py
|   |-- threshold_optimization.py
|   |-- shap_explain.py
|   |-- production_model.py
|   |-- predict.py
|   |-- transformers.py
|   |-- temporal_validation.py
|   |-- pretransaction_validation.py
|   `-- ablation_study.py
|
|-- tests/
|   |-- test_api.py
|   |-- test_model.py
|   `-- test_preprocessing.py
|
|-- Dockerfile.api
|-- Dockerfile.dashboard
|-- docker-compose.yml
|-- requirements.txt
|-- requirements-docker.txt
|-- .dockerignore
|-- .gitignore
`-- README.md
```

---

## Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Joblib

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Streamlit

### Testing

- Pytest

### DevOps

- Docker
- Docker Compose
- Git
- GitHub
- Railway

---

## Key Engineering Decisions

### Stratified Dataset Splitting

Stratified splitting preserves the extremely rare fraud distribution across training, validation, and test datasets.

### Temporal Validation

A chronological holdout was added to verify that strong model performance persisted when training on earlier transactions and evaluating on future transactions.

### Separate Validation and Test Sets

The validation set was used for model comparison and threshold analysis.

The final test set remained untouched until production evaluation.

### Metrics Appropriate for Imbalanced Data

Accuracy was not used as the primary performance metric.

The project emphasizes:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- False Negatives
- False Positives

### Feature Engineering Validation

The engineered-feature Random Forest was directly compared against a raw-feature Random Forest.

Feature ablation was later added to determine which engineered signals were responsible for the performance improvement.

### Leakage Analysis

Post-transaction features were removed in a separate experiment to test whether future information was responsible for the near-perfect results.

### Dataset Shortcut Analysis

The unusually strong `amount_to_orig_balance` signal was investigated statistically rather than treating the near-perfect benchmark performance as evidence of real-world generalization.

### Explainability

SHAP was integrated to provide visibility into model behavior.

### Production Pipeline

Feature engineering, preprocessing, and model inference are packaged into a reusable production pipeline.

### API / UI Separation

FastAPI handles model inference independently from the Streamlit interface.

This allows other applications to consume the fraud detection API.

### Containerization

Docker provides reproducible execution across local and cloud environments.

### Private Service Communication

The deployed Streamlit dashboard communicates with FastAPI using Railway's private network.

---

## Future Improvements

Possible extensions include:

- Evaluate the system on additional fraud datasets with different fraud-generation patterns
- Build a dedicated production pre-transaction model using only features available at authorization time
- Real-time transaction streaming
- Model monitoring
- Data drift detection
- Prediction logging
- Authentication and API keys
- Rate limiting
- Database integration
- CI/CD with GitHub Actions
- Automated retraining pipelines
- Cloud monitoring and alerting
- Additional fraud detection models

---

## Disclaimer

This is an educational machine learning portfolio project trained on the synthetic PaySim dataset.

The reported performance reflects evaluation on PaySim and should not be interpreted as expected performance on real-world financial transactions.

The robustness analysis identified a strong dataset-specific relationship between transaction amount and sender balance that contributes substantially to the near-perfect benchmark results.

The model should not be used for real financial or banking decisions without external validation on representative real-world data, production monitoring, security controls, regulatory review, and domain-specific evaluation.