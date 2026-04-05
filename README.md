# ML Netflix Artwork Optimization POC

A proof-of-concept ML system that predicts which thumbnail image will
drive the highest click-through rate for a given Netflix title and user
segment, inspired by Netflix's artwork personalization engine.

The system combines three data sources: Netflix content metadata,
aesthetic quality scores from the AVA photography dataset, and a
CTR prediction model trained on real user-ad interaction data.

## Notebooks

| Notebook | Description |
|---|---|
| `01_criteo_ctr_model.ipynb` | Initial CTR pipeline — data cleaning, feature engineering, Logistic Regression + XGBoost baseline |
| `02_criteo_ctr_eda.ipynb` | Exploratory analysis of the ad click dataset — class imbalance, feature distributions, click rate by segment |
| `03_netflix_eda.ipynb` | Netflix catalog EDA — content type, genre breakdown, titles added per year, missing values |
| `04_ava_unified_pipeline.ipynb` | AVA aesthetic scoring, genre mapping, unified feature table, CTR predictions, genre × segment heatmap |

## Results

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.5573 | 0.1436 | 0.6081 |
| XGBoost | 0.5814 | 0.1466 | 0.5891 |
| XGBoost (5-Fold CV) | 0.5836 ± 0.004 | — | — |

The 0.58 AUC reflects a meaningful ceiling for metadata-only features.
Adding visual quality signals (via AVA) is what moves this number —
which is the core finding of the project and the motivation for the
unified pipeline in notebook 04.

## Key Output

The system generates 26,421 predictions — one per Netflix title per
user segment (drama viewer, action viewer, family viewer). The
genre × segment heatmap shows where personalization has the most
impact. Thrillers show the starkest difference: action viewers are
predicted to click at 0.447 vs family viewers at 0.156.

Output files are in `outputs/`:
- `netflix_ctr_predictions.csv` — full predictions (26,421 rows)
- `dashboard_summary.csv` — avg CTR by genre × segment (heatmap data)
- `best_segment_per_title.csv` — winning segment per title

## Data

| Dataset | Source | Size | Role |
|---|---|---|---|
| Ad Click Prediction | [Kaggle — arashnic](https://www.kaggle.com/datasets/arashnic/ctr-in-advertisement) | 463,291 rows | CTR model training |
| Netflix Shows | [Kaggle — shivamb](https://www.kaggle.com/datasets/shivamb/netflix-shows) | 8,807 titles | Content catalog |
| AVA Aesthetic Visual Analysis | [Kaggle — nicolacarrassi](https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment) | 255,530 images | Visual quality scores |

Data files are not tracked. Place CSVs in `data/` if running locally.
Only `AVA.txt` and `tags.txt` are needed from the AVA dataset —
image downloads are not required.

## How to Reproduce

1. Open any notebook in Google Colab
2. Upload your `kaggle.json` API token when prompted
3. Run all cells in order

Run notebooks in sequence (01 → 02 → 03 → 04). Notebook 04 depends
on the trained XGBoost model from 01/02 — a self-contained retraining
cell is included at the top of 04 if running it standalone.

## Cloud Deployment Plan (AWS)

The next phase migrates this pipeline to AWS:
S3 (raw CSVs) → SageMaker (train + deploy endpoint)
→ Lambda (calls endpoint, writes predictions)
→ RDS MySQL (predictions table, verified via DBeaver)
→ Streamlit (dashboard — genre × segment heatmap)

Estimated cost: under $5 using AWS free tier + $50 course credits.

## Repository Structure
ml-netflix-artwork-optimization-poc/
├── notebooks/
│   ├── 01_criteo_ctr_model.ipynb
│   ├── 02_criteo_ctr_eda.ipynb
│   ├── 03_netflix_eda.ipynb
│   └── 04_ava_unified_pipeline.ipynb
├── figures/
│   ├── fig1_class_distribution.png
│   ├── fig2_missing_values.png
│   ├── fig3_feature_distributions.png
│   ├── fig4_click_rates_by_feature.png
│   ├── fig5_model_results.png
│   ├── fig6_feature_importance.png
│   ├── fig_netflix_eda.png
│   └── fig_predicted_ctr_heatmap.png
├── outputs/
│   ├── netflix_ctr_predictions.csv
│   ├── dashboard_summary.csv
│   └── best_segment_per_title.csv
├── models/
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── data/                          ← not tracked (see .gitignore)
├── .gitignore
└── README.md

## Requirements

All code runs in Google Colab — no local installation needed.
A Kaggle account and `kaggle.json` API token are required.

Key libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`imbalanced-learn`, `matplotlib`, `seaborn`