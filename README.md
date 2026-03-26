# ML Netflix Artwork Optimization POC

A machine learning system that predicts which thumbnail images drive 
higher click-through rates, inspired by Netflix's artwork personalization 
engine. This project uses a CTR prediction model trained on real user-ad 
interaction data as a proxy for thumbnail click behavior.

## Notebooks
- `notebooks/01_criteo_ctr_model.ipynb` — full CTR prediction pipeline 
  including data cleaning, feature engineering, SMOTE oversampling, 
  Logistic Regression baseline, XGBoost model, and 5-fold cross-validation

## Results
| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.5553 | 0.5571 | 0.1399 |
| XGBoost (SMOTE) | 0.6349 | 0.5762 | 0.1461 |
| XGBoost (5-Fold CV) | — | 0.6624 ± 0.011 | — |

## How to Reproduce
1. Open `notebooks/01_criteo_ctr_model.ipynb` in Google Colab
2. Upload your `kaggle.json` API token when prompted
3. The notebook will automatically download the dataset from Kaggle
4. Run all cells in order

Expected outputs:
- ROC-AUC ~0.66 (5-fold CV)
- Feature importance chart saved to `models/feature_importance.png`
- Confusion matrix saved to `models/confusion_matrix.png`

## Data
- **Ad Click Prediction dataset**: 
  kaggle.com/datasets/arashnic/ctr-in-advertisement  
  463,291 user-ad interaction records with binary click label.  
  Place in `data/` folder if running locally.

## Repository Structure
```
ml-netflix-artwork-optimization-poc/
├── notebooks/
│   └── 01_criteo_ctr_model.ipynb
├── models/
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── data/                          ← not tracked (see .gitignore)
└── README.md
```

## Requirements
All code runs in Google Colab — no local installation needed.
A Kaggle account and API token (`kaggle.json`) are required 
to download the dataset.