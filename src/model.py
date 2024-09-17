# model.py

import pickle
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    """
    Train a RandomForest model with the provided dataset.
    :param X: Features
    :param y: Target
    :return: Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, path):
    """
    Save the trained model to a file.
    :param model: Trained model
    :param path: Path to save the model
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Load a saved model from a file.
    :param path: Path to load the model from
    :return: Loaded model
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
