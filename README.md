# ML Netflix Artwork Optimization POC

A machine learning system that predicts which thumbnail images 
drive higher click-through rates, inspired by Netflix's artwork 
personalization engine.

## Notebooks
- `notebooks/01_ava_visual_quality_model.ipynb` — visual quality 
   classifier trained on the AVA dataset
- `notebooks/02_criteo_ctr_model.ipynb` — CTR prediction model 
   trained on the Criteo dataset

## Data
Download datasets from:
- AVA: https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment
- Criteo: https://www.kaggle.com/c/criteo-display-ad-challenge

Place downloaded files in the `data/` folder before running notebooks.