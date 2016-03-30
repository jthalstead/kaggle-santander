## kaggle-santander
R Script for submissions to Santander Customer Satisfaction competition.

Script Summary:
- Remove constant features
- Remove identical features
- Bin some variables
- XGBoost (binary:logistic)
  - Shrinkage: 0.02
  - Max Depth: 6
  - Subsample: 0.8 | 0.7
  - N Trees: 560
  - Objective: AUC
