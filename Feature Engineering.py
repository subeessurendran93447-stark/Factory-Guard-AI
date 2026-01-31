import pandas as pd
import numpy as np
import joblib

def engineer_factoryguard_features(df):
    """
    Inputs: df with ['timestamp', 'robot_id', 'vibration', 'temp', 'pressure']
    Output: Feature-rich dataframe for TSC
    """
    # Ensure time-series integrity
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['robot_id', 'timestamp'])
    
    # 500 Robots require grouped operations
    grouped = df.groupby('robot_id')
    
    sensors = ['vibration', 'temp', 'pressure']
    # Time windows (assuming 1-minute intervals: 60=1h, 360=6h, 720=12h)
    windows = {'1h': 60, '6h': 360, '12h': 720}

    for sensor in sensors:
        for name, size in windows.items():
            # 1. Rolling Mean: Captures the 'Level' of the signal
            df[f'{sensor}_mean_{name}'] = grouped[sensor].transform(lambda x: x.rolling(window=size).mean())
            
            # 2. Rolling Std Dev: Captures 'Volatility' (The most important failure indicator)
            df[f'{sensor}_std_{name}'] = grouped[sensor].transform(lambda x: x.rolling(window=size).std())
            
        # 3. EMA: Captures 'Recent Momentum' (weighted towards the now)
        df[f'{sensor}_ema_12h'] = grouped[sensor].transform(lambda x: x.ewm(span=720).mean())

        # 4. Lag Features: Captures 'Instantaneous Change'
        df[f'{sensor}_lag_1'] = grouped[sensor].shift(1)
        df[f'{sensor}_lag_2'] = grouped[sensor].shift(2)

    # Clean up and optimize memory for 500-robot scale
    df = df.dropna()
    fcols = df.select_dtypes('float').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    return df

# --- Saving the Engine Output ---
# processed_data = engineer_factoryguard_features(raw_iot_data)
# joblib.dump(processed_data, 'factoryguard_features.joblib', compress=3)