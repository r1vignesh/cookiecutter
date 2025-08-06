import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Loaded train data from {train_path} with shape {train_data.shape}")
        logging.info(f"Loaded test data from {test_path} with shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def extract_features(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    max_features: int = None
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Feature extraction with Bag of Words completed.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        raise

def save_features(
    X_train_bow, y_train, X_test_bow, y_test, output_dir: str = "data/interim"
) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['sentiment'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['sentiment'] = y_test
        train_path = os.path.join(output_dir, "train_bow.csv")
        test_path = os.path.join(output_dir, "test_bow.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info(f"Saved train features to {train_path}")
        logging.info(f"Saved test features to {test_path}")
    except Exception as e:
        logging.error(f"Failed to save features: {e}")
        raise

def main() -> None:
    try:
        params = load_params()
        max_features = params['feature_engg'].get('max_features', None)
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        X_train_bow, X_test_bow, _ = extract_features(X_train, X_test, max_features)
        save_features(X_train_bow, y_train, X_test_bow, y_test)
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
