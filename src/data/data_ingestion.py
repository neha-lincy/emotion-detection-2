import os
import yaml
import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

# Ensure the 'logs' directory exists (absolute path)
log_dir = 'D:/emotiondetection/logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'data_ingestion.log'),
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
        logging.error("YAML file not found: %s", file_path)
        raise e
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML file: %s", file_path)
        raise e

def load_dataset(url: str) -> pd.DataFrame:
    """
    Load dataset from a given URL.

    Args:
        url (str): URL of the dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully from %s", url)
        return df
    except Exception as e:
        logging.error("Error loading dataset from %s", url)
        raise e

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by dropping unnecessary columns and filtering rows.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    try:
        # Drop the 'tweet_id' column
        df.drop(columns=['tweet_id'], inplace=True)

        # Filter rows with sentiment 'happiness' or 'sadness'
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        # Replace sentiment labels with numerical values
        df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

        logging.info("Dataset preprocessing completed successfully")
        return df
    except KeyError as e:
        logging.error("Error during preprocessing: Missing column %s", e)
        raise e
    except Exception as e:
        logging.error("Unexpected error during preprocessing")
        raise e

def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
    """
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info("Data split into training and testing sets successfully")
        return train_data, test_data
    except Exception as e:
        logging.error("Error during data splitting")
        raise e

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """
    Save the training and testing datasets to the specified directory.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        output_dir (str): Directory to save the datasets.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        logging.info("Training and testing datasets saved successfully in %s", output_dir)
    except Exception as e:
        logging.error("Error saving datasets to %s", output_dir)
        raise e

def main() -> None:
    """
    Main function to execute the data ingestion pipeline.
    """
    try:
        # Load parameters
        params = load_params("params.yaml")
        test_size = params['data_ingestion']['test_size']

        # Load and preprocess the dataset
        raw_data = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        processed_data = preprocess_data(raw_data)

        # Split the data
        train_data, test_data = split_data(processed_data, test_size=test_size, random_state=42)

        # Save the data
        save_data(train_data, test_data, "data/raw")

        logging.info("Data ingestion pipeline executed successfully")
    except Exception as e:
        logging.error("Data ingestion pipeline failed")
        raise e

if __name__ == "__main__":
    main()
