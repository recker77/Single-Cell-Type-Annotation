# Single-Cell Type Annotation with XGBoost and CatBoost

## Overview
This repository provides an approach to annotate single-cell data types using machine learning models, **XGBoost** and **CatBoost**. Single-cell type annotation helps understand cellular heterogeneity in biological systems.

## Features
- Preprocessing pipeline for normalization and feature selection.
- Model training with XGBoost and CatBoost.
- Feature importance analysis.
- Evaluation using metrics like accuracy, F1-score, and ROC-AUC.

## Prerequisites
### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Data
Prepare a single-cell dataset in CSV format with rows as cells, columns as features, and the last column as cell type labels.

## Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/recker77/single-cell-annotation.git
   cd single-cell-annotation
   ```

2. **Prepare Your Dataset:**
   Place your dataset in the `data/` folder (e.g., `single_cell_data.csv`).

3. **Run the Pipeline:**
   ```bash
   python main.py --data data/single_cell_data.csv --output results/
   ```

4. **Results:**
   Outputs in `results/` include trained model files, performance metrics, feature importance plots, and confusion matrices.

## Repository Structure
```plaintext
.
|-- data/                # Input dataset
|-- results/             # Output folder
|-- src/                 # Source code
|-- main.py              # Entry point
|-- requirements.txt     # Dependencies
```

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

