import joblib
import pandas as pd
import shap
import numpy as np
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    average_precision_score, 
    precision_recall_curve,
    f1_score,
    PrecisionRecallDisplay
)

df = pd.read_csv("Factory Guard.csv")

target = "failure_in_next_24h" 
X = df.drop(columns=[target, "timestamp", "torque_nm", "motor_current_a", "rpm"])
y = df[target]

train_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]

num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_weight = num_neg / num_pos

print(f"Imbalance Ratio: 1:{scale_weight:.2f}")

xgb_model = XGBClassifier(scale_pos_weight=scale_weight, eval_metric='aucpr')

xgb_model.fit(X_train, y_train)

y_probs = xgb_model.predict_proba(X_test)[:, 1]
y_preds = xgb_model.predict(X_test)

pr_auc = average_precision_score(y_test, y_probs)

print("\n--- Production XGBoost Model Results ---")
print(f"PR-AUC Score: {pr_auc:.4f}")
print(f"F1-Score: {f1_score(y_test, y_preds):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_preds))
print("\nClassification Report:")
print(classification_report(y_test, y_preds))

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.title("SHAP Feature Impact on Failure Prediction")
shap.summary_plot(shap_values, X_test)

sample_idx = 0 
plt.figure(figsize=(10, 5))
explanation = shap.Explanation(
    values=shap_values[sample_idx], 
    base_values=explainer.expected_value, 
    data=X_test.iloc[sample_idx], 
    feature_names=X.columns
)

shap.plots.waterfall(explanation)

vals = explanation.values
feature_names = X.columns
top_indicators = pd.Series(vals, index=feature_names).sort_values(ascending=False).head(3)

print(f"Maintenance Alert for Unit {sample_idx}:")
for feat, val in top_indicators.items():
    print(f"- {feat} is increasing failure risk (SHAP value: {val:.2f})")