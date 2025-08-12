import pickle
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import Dict
import logging

# Configure logging
logging.basicConfig(
    filename="logs/model_evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully from %s", file_path)
        return model
    except Exception as e:
        logging.error("Error loading model: %s", e)
        raise e

def load_test_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logging.info("Test data loaded from %s", file_path)
        return data
    except Exception as e:
        logging.error("Error loading test data: %s", e)
        raise e

def evaluate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    try:
        y_pred = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        logging.info("Model evaluation completed")
        return metrics
    except Exception as e:
        logging.error("Error during evaluation: %s", e)
        raise e

def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info("Evaluation metrics saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving metrics: %s", e)
        raise e

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")

        x_test = test_data.drop(columns=['sentiment']).values
        y_test = test_data['sentiment'].values

        metrics = evaluate_model(model, x_test, y_test)
        save_metrics(metrics, "reports/evaluation_metrics.json")
    except Exception as e:
        logging.error("Model evaluation pipeline failed: %s", e)
        raise e

if __name__ == "__main__":
    main()
