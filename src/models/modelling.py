import os
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

# Ensure 'logs' directory exists
log_dir = 'D:/emotiondetection/logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging with absolute path for the log file
logging.basicConfig(
    filename=os.path.join(log_dir, "modelling.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params(file_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing parameters.
    """
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", file_path)
        return params
    except FileNotFoundError as e:
        logging.error("Parameters file not found: %s", e)
        raise e
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML file: %s", e)
        raise e
    except Exception as e:
        logging.error("Unexpected error while loading parameters: %s", e)
        raise e

def load_training_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed training data from a CSV file.

    Args:
        file_path (str): Path to the training data file.

    Returns:
        pd.DataFrame: Loaded training data.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Training data loaded successfully from %s", file_path)
        return data
    except FileNotFoundError as e:
        logging.error("Training data file not found: %s", e)
        raise e
    except Exception as e:
        logging.error("Error loading training data from %s: %s", file_path, e)
        raise e

def separate_features_and_labels(data: pd.DataFrame, label_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate features and labels from the training data.

    Args:
        data (pd.DataFrame): Input DataFrame containing features and labels.
        label_column (str): Column name for the labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y).
    """
    try:
        x = data.drop(columns=[label_column]).values
        y = data[label_column].values
        logging.info("Features and labels separated successfully")
        return x, y
    except KeyError as e:
        logging.error("Missing column in DataFrame: %s", e)
        raise e
    except Exception as e:
        logging.error("Error separating features and labels: %s", e)
        raise e

def train_random_forest(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logging.error("Error training Random Forest model: %s", e)
        raise e

def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """
    Save the trained model to a file using pickle.

    Args:
        model (RandomForestClassifier): Trained model.
        file_path (str): Path to save the model file.
    """
    try:
        if os.path.exists(file_path):
            logging.warning("Model file already exists at %s, it will be overwritten.", file_path)
        
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logging.info("Model saved successfully to %s", file_path)
    except Exception as e:
        logging.error("Error saving model to %s: %s", file_path, e)
        raise e

def main() -> None:
    """
    Main function to execute the modelling pipeline.
    """
    try:
        # Load parameters
        params = load_params("params.yaml")
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']

        # Load training data
        train_data = load_training_data("data/interim/train_bow.csv")

        # Separate features and labels
        x_train, y_train = separate_features_and_labels(train_data, label_column='sentiment')

        # Train the Random Forest model
        model = train_random_forest(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth)

        # Save the trained model
        save_model(model, "models/random_forest_model.pkl")

        logging.info("Modelling pipeline executed successfully")
    except Exception as e:
        logging.error("Modelling pipeline failed: %s", e)
        raise e

if __name__ == "__main__":
    main()
