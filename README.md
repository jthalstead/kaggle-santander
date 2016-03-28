## kaggle-santander
R Script for submissions to Santander Customer Satisfaction competition.

Script Summary:
- Remove constant features
- Remove identical features
- XGBoost (binary:logistic)
  - Shrinkage: 0.02
  - Max Depth: 6
  - Subsample: 0.9 | 0.85
  - N Trees: 500
  - Objective: AUC
