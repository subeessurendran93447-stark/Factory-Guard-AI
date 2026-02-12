import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("factoryguard_synthetic_500.csv")
print(df.head(10))
df.time_to_failure_hours = df.time_to_failure_hours.fillna(df.time_to_failure_hours.mean())
df = df.drop(columns=["humidity_pct", "age_hours", "maintenance_days_ago", "error_count", "load_pct", "time_to_failure_hours"])
print(df.info())
print(df.columns)
print(df.isnull().sum())
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(df)
sensor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(sensor_cols)
if "id" in sensor_cols: sensor_cols.remove('id')
windows = [1, 6, 12]
for sensor in sensor_cols:
    for w in windows:
    
        df[f'{sensor}_roll_mean_{w}h'] = df[sensor].rolling(window=w).mean()
        
        df[f'{sensor}_roll_std_{w}h'] = df[sensor].rolling(window=w).std()
        
        df[f'{sensor}_ema_{w}h'] = df[sensor].ewm(span=w, adjust=False).mean()

    df[f'{sensor}_lag_1'] = df[sensor].shift(1)

    df[f'{sensor}_lag_2'] = df[sensor].shift(2)

df = df.groupby('arm_id').apply(lambda x: x.iloc[12:]).reset_index(drop=True)
# 2. Save the DataFrame as a .joblib file
# We use compress=3 as a good balance between speed and file size
joblib.dump(df, "FactoryGuard.joblib", compress=3)

print("Conversion complete!")

df.to_csv("Factory Guard.csv",index = False)
