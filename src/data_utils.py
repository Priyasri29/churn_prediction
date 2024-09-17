# data_utils.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Generic function to load CSV data."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Generic function to preprocess different datasets for churn prediction."""
    # Handle missing values
    data = data.fillna('Unknown')

    # Convert categorical variables using LabelEncoder
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Ensure the target column ('churn') is present
    if 'churn' not in data.columns:
        raise ValueError("The dataset must contain a 'churn' column as the target variable.")
    
    return data, label_encoders
