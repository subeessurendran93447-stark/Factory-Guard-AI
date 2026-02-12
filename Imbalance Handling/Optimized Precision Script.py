import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline

df = pd.read_csv("Factory Guard.csv")
target = "failure_in_next_24h"

X = df.drop(columns=[target, "timestamp", "torque_nm", "motor_current_a", "rpm"])
y = df[target]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])

enn = EditedNearestNeighbours(sampling_strategy='auto')
xgb_model = XGBClassifier(n_estimators=100, reg_lambda=15, learning_rate=0.05)

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('resampler', enn),
    ('classifier', xgb_model)
])

train_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]

clf_pipeline.fit(X_train, y_train)

y_probs = clf_pipeline.predict_proba(X_test)[:, 1]
y_preds = (y_probs >= 0.8).astype(int)


print(f"Precision: {precision_score(y_test, y_preds):.4f}")
print(classification_report(y_test, y_preds))