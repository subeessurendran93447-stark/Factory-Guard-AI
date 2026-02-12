import joblib
import pandas as pd
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

xgb_model = XGBClassifier()

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


# --- Visualizations ---


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(8, 6))
display = PrecisionRecallDisplay.from_predictions(y_test, y_probs, name="XGBoost")
_ = display.ax_.set_title("2nd-Generation Precision-Recall Curve")
plt.show()


plt.figure(figsize=(10, 8))
plot_importance(xgb_model, importance_type='gain', max_num_features=10)
plt.title('Top 10 Feature Importance (Gain)')
plt.show()