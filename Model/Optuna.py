import joblib
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("Factory Guard.csv")
print(df.info())

target = "failure_in_next_24h" 
X = df.drop(columns=[target,"timestamp","torque_nm","motor_current_a","rpm"])
y = df[target]

train_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]

y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]

num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_weight = num_neg / num_pos

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        # Using the scale_weight you calculated earlier
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_weight * 0.8, scale_weight * 1.2),
        'use_label_encoder': False,
        'eval_metric': 'aucpr' 
    }

    model = XGBClassifier()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)

print(f"Best PR-AUC: {study.best_value:.4f}")
print("Best Params:", study.best_params)

# Train the final model with best parameters
best_model = XGBClassifier()
best_model.fit(X_train, y_train)

# Run your evaluation again
y_probs_optimized = best_model.predict_proba(X_test)[:, 1]
y_preds_optimized = best_model.predict(X_test)

print(f"Optimized PR-AUC: {average_precision_score(y_test, y_probs_optimized):.4f}")