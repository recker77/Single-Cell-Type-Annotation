# Single-Cell Type Annotation with XGBoost and CatBoost

## Overview
This repository provides an approach to annotate single-cell data types using machine learning models, **XGBoost** and **CatBoost**. Single-cell type annotation helps understand cellular heterogeneity in biological systems. There were 61 type of cells to annotate.

## Features
- Preprocessing pipeline for normalization and feature selection.
- Model training with XGBoost and CatBoost.
- Feature importance analysis.
- Evaluation using metrics like accuracy, macro avg, and weighted avg.

## Prerequisites
### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Data
The data has 162709 rows.

## Model Details
- **XGBoost**: Gradient Boosting, tuned with Optuna.
- **CatBoost**: Ordered Boosting, efficient handling of categorical data.

## Performance
- **Accuracy**: XGBoost: 81.34%, CatBoost: 79.83%
- **Macro Avg**: XGBoost: 78.12%, CatBoost: 77.21%
- **Weighted Avg**: XGBoost: 0.81, CatBoost: 0.79

## License
This project is licensed under the MIT License.

## Acknowledgments
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

