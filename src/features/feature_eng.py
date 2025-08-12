import os
import yaml
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(
    filename="logs/feature_eng.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", file_path)
        return params
    except Exception as e:
        logging.error("Error loading parameters: %s", e)
        raise e

def load_data(file_path: str, drop_column: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path).dropna(subset=[drop_column])
        logging.info("Data loaded from %s", file_path)
        return data
    except Exception as e:
        logging.error("Error loading data from %s: %s", file_path, e)
        raise e

def extract_features_and_labels(data: pd.DataFrame, feature_column: str, label_column: str) -> Tuple[pd.Series, pd.Series]:
    try:
        features = data[feature_column].values
        labels = data[label_column].values
        return features, labels
    except Exception as e:
        logging.error("Error extracting features and labels: %s", e)
        raise e

def initialize_vectorizer(max_features: int) -> TfidfVectorizer:
    try:
        return TfidfVectorizer(max_features=max_features)
    except Exception as e:
        logging.error("Error initializing TfidfVectorizer: %s", e)
        raise e

def save_vectorized_data(features, labels, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(features.toarray())
        df['sentiment'] = labels
        df.to_csv(output_path, index=False)
        logging.info("Vectorized data saved to %s", output_path)
    except Exception as e:
        logging.error("Error saving vectorized data: %s", e)
        raise e

def main() -> None:
    try:
        params = load_params("params.yaml")
        max_features = params['feature_eng']['max_features']

        # Load processed data
        train_data = load_data("data/processed/train.csv", drop_column='content')
        test_data = load_data("data/processed/test.csv", drop_column='content')

        # Extract features and labels
        x_train, y_train = extract_features_and_labels(train_data, 'content', 'sentiment')
        x_test, y_test = extract_features_and_labels(test_data, 'content', 'sentiment')

        # Vectorize
        vectorizer = initialize_vectorizer(max_features)
        x_train_vectorized = vectorizer.fit_transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        # Save interim data
        save_vectorized_data(x_train_vectorized, y_train, "data/interim/train_tfidf.csv")
        save_vectorized_data(x_test_vectorized, y_test, "data/interim/test_tfidf.csv")

        logging.info("Feature engineering completed")
    except Exception as e:
        logging.error("Feature engineering pipeline failed: %s", e)
        raise e

if __name__ == "__main__":
    main()
