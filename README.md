# Single-Cell Type Annotation with XGBoost and CatBoost

## Overview
This repository provides a streamlined approach to annotate single-cell data types using powerful machine learning models, **XGBoost** and **CatBoost**. Single-cell type annotation is a critical step in understanding cellular heterogeneity in biological systems. This repository demonstrates how to effectively use gradient boosting algorithms to classify cell types from single-cell datasets.

## Features
- **Preprocessing Pipeline**: Tools for handling single-cell data, including normalization and feature selection.
- **Model Training**: Implementation of XGBoost and CatBoost models optimized for cell type classification.
- **Feature Importance Analysis**: Insights into the most influential features contributing to predictions.
- **Evaluation Metrics**: Performance evaluation using accuracy, F1-score, ROC-AUC, and confusion matrices.

## Prerequisites
### Dependencies
The following Python packages are required:
- Python >= 3.8
- pandas
- numpy
- scikit-learn
- xgboost
- catboost
- matplotlib

You can install the required packages using:
```bash
pip install -r requirements.txt
```

### Data
Prepare a single-cell dataset in CSV format where rows represent cells, columns represent features, and the last column represents the cell type labels.

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/single-cell-annotation.git
cd single-cell-annotation
```

### 2. Prepare Your Dataset
Place your dataset in the `data/` folder and ensure it’s named appropriately (e.g., `single_cell_data.csv`).

### 3. Run the Pipeline
Execute the main script to preprocess data, train models, and evaluate performance:
```bash
python main.py --data data/single_cell_data.csv --output results/
```

### 4. Results
The script generates the following outputs in the `results/` directory:
- Trained model files (`xgboost_model.pkl`, `catboost_model.pkl`)
- Performance metrics (accuracy, F1-score, ROC-AUC, etc.)
- Feature importance plots
- Confusion matrices

## Repository Structure
```plaintext
.
|-- data/                # Input dataset folder
|-- results/             # Output folder for results and models
|-- src/                 # Source code
|   |-- preprocess.py    # Data preprocessing utilities
|   |-- train.py         # Model training and evaluation scripts
|   |-- utils.py         # Helper functions
|-- main.py              # Entry point for running the pipeline
|-- requirements.txt     # Python dependencies
|-- README.md            # Project documentation
```

## Model Details
### XGBoost
- Boosting Type: Gradient Boosting
- Evaluation Metric: Multi-class Log Loss
- Features: Tuned hyperparameters using optuna for optimal performance.

### CatBoost
- Boosting Type: Ordered Boosting
- Evaluation Metric: Multi-class Log Loss
- Features: Handles categorical data efficiently without preprocessing.

## Performance
Both models were evaluated on a benchmark single-cell dataset:
- **Accuracy**: XGBoost: 81.34%, CatBoost: 79.83%
- **Macro Avg**: XGBoost: 78.12%, CatBoost: 77.21%
- **Weighted Avg**: XGBoost: 0.81, CatBoost: 0.79

## Customization
You can modify the training configurations in the `config/` folder to adapt the pipeline to your specific dataset or requirements.

## Contributions
Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- Open-source single-cell datasets used for validation.

