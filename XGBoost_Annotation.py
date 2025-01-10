# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import joblib

data = pd.read_csv("/home/raghvendra/home3/raghvendra/Crunch_2/Sumit/data.csv")
data.head()

data.drop(['status'],axis=1, inplace=True)
data.head()

# +
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]


unique_labels = sorted(Y.unique())
label_encoder = LabelEncoder()
label_encoder.fit(unique_labels)


Y_encoded = label_encoder.transform(Y)


joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder for target saved successfully!")


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)


X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()


for col in X.select_dtypes(include=['object']).columns:
    X_train_encoded[col] = X_train_encoded[col].astype(str)
    X_test_encoded[col] = X_test_encoded[col].astype(str)
    
    unique_values = sorted(X_train_encoded[col].unique())
    
    le = LabelEncoder()
    le.fit(unique_values)
    
    X_train_encoded[col] = le.transform(X_train_encoded[col])
    X_test_encoded[col] = X_test_encoded[col].apply(
        lambda val: le.transform([val])[0] if val in unique_values else -1
    )
    
    joblib.dump(le, f'{col}_label_encoder.pkl')
    print(f"Label encoder for column '{col}' saved successfully!")


# -

def objective(trial):
    params = {
        'objective': 'multi:softmax',
        'num_class': len(unique_labels),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 0, 10),
        'seed': 42,
        'nthread':60
    }
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, Y_train)):
        X_train_fold, X_val_fold = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]
        
        dtrain = xgb.DMatrix(X_train_fold, label=Y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=Y_val_fold)
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=[(dval, 'validation')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        accuracy = accuracy_score(Y_val_fold, preds)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    print(f"Trial {trial.number}: Mean Accuracy = {mean_accuracy:.4f}")
    return mean_accuracy


# +
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# +
best_params = study.best_params
best_params.update({
    'objective': 'multi:softmax',
    'num_class': len(unique_labels),
    'seed': 42
})


dtrain = xgb.DMatrix(X_train_encoded, label=Y_train)
final_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=study.best_trial.user_attrs.get('best_iteration', 500)
)


joblib.dump(final_model, 'best_xgboost_model.pkl')
print("Final model saved successfully!")

# +
dtest = xgb.DMatrix(X_test_encoded)
Y_pred = final_model.predict(dtest)

Y_pred_decoded = label_encoder.inverse_transform(Y_pred.astype(int))
Y_test_decoded = label_encoder.inverse_transform(Y_test)

# +
accuracy = accuracy_score(Y_test_decoded, Y_pred_decoded)
macro_f1 = f1_score(Y_test_decoded, Y_pred_decoded, average='macro')
precision = precision_score(Y_test_decoded, Y_pred_decoded, average='macro')
recall = recall_score(Y_test_decoded, Y_pred_decoded, average='macro')
conf_matrix = confusion_matrix(Y_test_decoded, Y_pred_decoded)

#
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)