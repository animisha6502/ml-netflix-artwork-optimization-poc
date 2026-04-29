# ML Netflix Artwork Optimization POC

A proof-of-concept ML system that predicts which thumbnail image will
drive the highest click-through rate for a given Netflix title and user
segment, inspired by Netflix's artwork personalization engine.

The system combines three data sources: Netflix content metadata,
aesthetic quality scores from the AVA photography dataset, and a
CTR prediction model trained on real user-ad interaction data.

## Live Demo

The Streamlit app has three features:
- **Thumbnail Scorer** — upload any image; returns predicted CTR, adjusted AVA score, face detection, and VADER sentiment breakdown
- **TMDB Image Ranker** — fetch all posters for any title via the TMDB API, score and rank by predicted CTR
- **A/B Simulator** — compare two titles head-to-head, project monthly clicks by segment

Run locally:
```bash
pip install -r requirements.txt
export TMDB_API_KEY=your_tmdb_key
streamlit run streamlit_app_final.py
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_criteo_ctr_model.ipynb` | CTR pipeline — data cleaning, feature engineering, Logistic Regression + XGBoost baseline |
| `02_criteo_ctr_eda.ipynb` | EDA of the ad click dataset — class imbalance, feature distributions, click rate by segment |
| `03_netflix_eda.ipynb` | Netflix catalog EDA — content type, genre breakdown, titles added per year, missing values |
| `04_ava_unified_pipeline.ipynb` | AVA aesthetic scoring, genre mapping, unified feature table, CTR predictions, genre × segment heatmap |
| `05_pipeline_test.ipynb` | End-to-end pipeline validation — confirms model outputs and prediction counts |

Run in sequence (01 → 02 → 03 → 04 → 05). Notebook 04 depends on the
trained XGBoost model from 01 — a self-contained retraining cell is
included at the top of 04 if running standalone.

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

**Colab:**
1. Open any notebook in Google Colab
2. Upload your `kaggle.json` API token when prompted
3. Run all cells in order (01 → 02 → 03 → 04 → 05)

**Locally:**
```bash
pip install -r requirements.txt
# Place data CSVs in data/ then run notebooks in order
```

## Repository Structure

```
ml-netflix-artwork-optimization-poc/
├── notebooks/
│   ├── 01_criteo_ctr_model.ipynb
│   ├── 02_criteo_ctr_eda.ipynb
│   ├── 03_netflix_eda.ipynb
│   ├── 04_ava_unified_pipeline.ipynb
│   └── 05_pipeline_test.ipynb
├── models/
│   ├── lr_ctr_model.pkl
│   ├── xgb_ctr_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── encoding_maps.json
├── figures/
│   ├── fig1_class_distribution.png
│   ├── fig2_missing_values.png
│   ├── fig3_feature_distributions.png
│   ├── fig4_click_rates_by_feature.png
│   ├── fig5_model_results.png
│   ├── fig6_feature_importance.png
│   ├── fig_netflix_eda.png
│   ├── fig_correlation_heatmap.png
│   ├── fig_missing_combined.png
│   ├── fig_temporal_patterns.png
│   ├── fig_predicted_ctr_heatmap.png
│   ├── test_ctr_distributions.png
│   └── test_genre_heatmap.png
├── outputs/
│   ├── netflix_ctr_predictions.csv
│   ├── dashboard_summary.csv
│   └── best_segment_per_title.csv
├── data/                          ← not tracked (see .gitignore)
├── streamlit_app_final.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

```
streamlit
pandas
numpy
matplotlib
Pillow
requests
opencv-python-headless
nltk
```

Notebooks also require: `scikit-learn`, `xgboost`, `imbalanced-learn`, `seaborn`.
The Streamlit app runs fully locally — no LLM or external API key needed
(TMDB key is optional and only required for the Image Ranker tab).
