import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_train_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded training data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_model(
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int, 
    max_depth: int
) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(x_train, y_train)
        logging.info("RandomForest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        raise

def save_model(model: RandomForestClassifier, path: str) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    try:
        params = load_params()
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']
        train_data = load_train_data("data/interim/train_bow.csv")
        x_train = train_data.drop(columns=['sentiment']).values
        y_train = train_data['sentiment'].values
        model = train_model(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Modelling pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Modelling pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
