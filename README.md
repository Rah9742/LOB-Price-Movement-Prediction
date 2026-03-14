# Limit Order Book Mid-Price Movement Prediction

Machine learning models for predicting **mid-price movement** in a limit order book (LOB) using the **FI-2010 benchmark dataset**.

The project evaluates a progression of models, from linear baselines to deep sequential models, to measure how increasing model complexity and temporal awareness improves predictive performance.

The models are trained and evaluated on **FI-2010 folds 7, 8, and 9**, using **NoAuction ZScore normalisation** and **prediction horizon k = 5**.

A full methodological description and experimental results are available in the project documentation. See: **reports/Summary.docx**.


# Environment Setup

Create and use a local virtual environment in the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project has been tested with **Python 3.13** in `.venv`.

If your existing `.venv` was created with global packages enabled, recreate it so the environment stays isolated:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To reactivate the environment later:

```bash
source .venv/bin/activate
```

On macOS, `xgboost` also requires the OpenMP runtime:

```bash
brew install libomp
```


# Project Overview

The task is to predict the **direction of the mid-price movement** in a limit order book over the next **k = 5 events**.

Each observation consists of **144 features** derived from order book levels, volumes, and derived statistics.

Classes:

| Label | Meaning |
|-----|-----|
| 1 | Up |
| 2 | Stationary |
| 3 | Down |

The dataset is **highly class-imbalanced**, with the Stationary class dominating. Therefore the primary evaluation metric is **Macro F1 score**, which treats all classes equally.


# Implemented Models

Six models are implemented, ordered from simplest to most complex.

| Model | Type | Input Representation |
|-----|-----|-----|
| Ridge Regression | Linear | 144 flat features |
| Logistic Regression | Linear classifier | 144 flat features |
| Multi-Layer Perceptron (MLP) | Feedforward neural network | 144 flat features |
| Random Forest | Tree ensemble | 144 flat features |
| XGBoost | Gradient boosted trees | 144 flat features |
| LSTM | Recurrent neural network | 100 × 144 sequences |

The progression allows comparison between:

- Linear models
- Non-linear tabular models
- Sequential deep learning models


# Repository Structure

```
LOB-Price-Movement-Prediction
│
├── data/
│   ├── Train_Dst_NoAuction_ZScore_CF_7.txt
│   ├── Train_Dst_NoAuction_ZScore_CF_8.txt
│   ├── Train_Dst_NoAuction_ZScore_CF_9.txt
│   ├── Test_Dst_NoAuction_ZScore_CF_7.txt
│   ├── Test_Dst_NoAuction_ZScore_CF_8.txt
│   └── Test_Dst_NoAuction_ZScore_CF_9.txt
│
├── reports/
│   ├── results_all_models.csv
│   ├── results_logistic.csv
│   ├── results_lstm.csv
│   ├── results_mlp.csv
│   ├── results_random_forest.csv
│   ├── results_ridge.csv
│   ├── results_xgboost.csv
│   └── Summary.docx
│
├── src/
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── logistic_regression.py
│   ├── lstm_model.py
│   ├── mlp_model.py
│   ├── random_forest.py
│   ├── ridge_regression.py
│   └── xgboost_model.py
│
├── .gitignore
└── README.md
```


# Dataset

The project uses the **FI-2010 benchmark dataset**, which contains event-driven snapshots of limit order books for five Finnish stocks.

Each snapshot contains **144 engineered features**, grouped into nine feature blocks representing:

- Order book price levels
- Volume levels
- Derived order book statistics
- Temporal changes in liquidity

The dataset is split into **nine folds**, each roughly corresponding to a trading day.


# Train/Test Protocol

Experiments follow an **anchored forward cross-validation scheme**:

For fold **N**:

Training data  
```
folds 1 → N-1
```

Testing data  
```
fold N
```

A validation split is carved from the end of the training data to perform:

- hyperparameter tuning
- early stopping


# Evaluation Metrics

The primary evaluation metric is:

**Macro F1 Score**

```
F1_macro = (F1_up + F1_stationary + F1_down) / 3
```

This metric is preferred because the dataset is class-imbalanced.

Additional reported metrics include:

- Accuracy
- Macro Precision
- Macro Recall
- Per-class F1 scores


# Model Highlights

## Linear Models
Ridge Regression and Logistic Regression act as **baseline models**. They treat each snapshot independently and cannot capture temporal dynamics.

## Neural Network (MLP)
The MLP learns **non-linear interactions between the 144 features**, improving performance over linear methods.

## Tree Ensembles
Random Forest and XGBoost model complex feature interactions without explicit feature engineering. XGBoost typically achieves stronger performance due to its **gradient-boosted sequential tree construction**.

## LSTM
The LSTM processes **100-step sequences of LOB snapshots**, enabling the model to learn temporal patterns in order book dynamics.

This temporal modelling yields a substantial performance improvement over static models.


# Example Results

Average performance across folds **7, 8, 9**.

| Model | Macro F1 | Accuracy |
|-----|-----|-----|
| Ridge Regression | 0.437 | 0.466 |
| Logistic Regression | 0.439 | 0.468 |
| MLP | 0.487 | 0.513 |
| Random Forest | 0.552 | 0.589 |
| XGBoost | 0.564 | 0.593 |
| LSTM | 0.765 | 0.782 |

The results demonstrate the strong value of **temporal modelling** for limit order book prediction.


# Running the Models

Example workflow:

```bash
source .venv/bin/activate
python -m src.ridge_regression
python -m src.logistic_regression
python -m src.mlp_model
python -m src.random_forest
python -m src.xgboost_model
python -m src.lstm_model
```

Each script:

1. Loads data using `data_loader.py`
2. Trains the model
3. Evaluates using `evaluator.py`
4. Saves results to `reports/`


# Hardware Acceleration

The LSTM model automatically detects the best available device:

- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon GPUs)
- **CPU fallback**

This is handled through PyTorch device selection.
