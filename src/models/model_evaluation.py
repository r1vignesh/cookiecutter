
import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_model(model_path: str) -> Any:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(test_path)
        logging.info(f"Test data loaded from {test_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics calculated: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Evaluation metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation metrics: {e}")
        raise

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        X_test = test_data.drop(columns=['sentiment']).values
        y_test = test_data['sentiment'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/evaluation_metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
