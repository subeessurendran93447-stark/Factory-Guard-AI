import joblib
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

df = pd.read_csv("Factory Guard.csv")
print(df.info())

target = "failure_in_next_24h" 
X = df.drop(columns=[target,"timestamp","torque_nm","motor_current_a","rpm"])
y = df[target]

train_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]

y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]


print("--- Training Baseline: Random Forest ---")
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)

rf_preds = rf_model.predict(X_test)
print(f"Baseline Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"Baseline F1-Score: {f1_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

print("--- Training Production XGBoost Model ---")

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
print(f"XGBoost F1-Score: {f1_score(y_test, xgb_preds):.4f}")
print(classification_report(y_test, xgb_preds))