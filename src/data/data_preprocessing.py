import os
import pandas as pd
import logging
from typing import Any

# Ensure the 'logs' directory exists
log_dir = 'D:/emotiondetection/logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging with absolute path for the log file
logging.basicConfig(
    filename=os.path.join(log_dir, "data_preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def lemmatization(sentence: str) -> str:
    """
    Perform lemmatization on a given sentence.

    Args:
        sentence (str): Input sentence to lemmatize.

    Returns:
        str: Lemmatized sentence.
    """
    try:
        # Placeholder for actual lemmatization logic
        # Replace this with your lemmatization implementation
        return sentence.lower()  # Example: converting to lowercase
    except Exception as e:
        logging.error("Error during lemmatization: %s", e)
        raise e

def normalized_sentence(sentence: str) -> str:
    """
    Apply all preprocessing steps to a single sentence.

    Args:
        sentence (str): Input sentence to preprocess.

    Returns:
        str: Preprocessed sentence.
    """
    try:
        # Placeholder for preprocessing steps
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error("Error in normalizing sentence: %s", e)
        raise e

def normalize_text(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text data in a DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing text data.

    Returns:
        pd.DataFrame: DataFrame with normalized text.
    """
    try:
        # Assuming the text column is named 'text'
        data['content'] = data['content'].apply(normalized_sentence)
        logging.info("Text normalization completed successfully")
        return data
    except KeyError as e:
        logging.error("Missing 'text' column in the DataFrame: %s", e)
        raise e
    except Exception as e:
        logging.error("Error in normalizing text data: %s", e)
        raise e

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logging.info("Data saved successfully to %s", file_path)
    except Exception as e:
        logging.error("Error saving data to %s: %s", file_path, e)
        raise e

def main() -> None:
    """
    Main function to execute the data preprocessing pipeline.
    """
    try:
        # Load raw train and test data
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logging.info("Raw train and test data loaded successfully")

        # Normalize train and test data
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        # Save processed data to CSV files
        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")

        logging.info("Data preprocessing pipeline executed successfully")
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise e
    except Exception as e:
        logging.error("Data preprocessing pipeline failed: %s", e)
        raise e

if __name__ == "__main__":
    main()
