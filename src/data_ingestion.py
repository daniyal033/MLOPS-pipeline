import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

# Create logs directory
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load data
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

# Preprocess data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, errors='ignore')
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataset: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise

# Save train and test data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        row_data_path = os.path.join(data_path, 'raw')
        os.makedirs(row_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(row_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(row_data_path, 'test.csv'), index=False)
        logger.debug("Train and test data saved to %s", row_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise

# Main function
def main():
    try:
        test_size = 0.2
        data_url = 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        data_path = './data'

        # Load and preprocess data
        df = load_data(data_url)
        final_df = preprocess(df)

        # Split data
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save data
        save_data(train_data, test_data, data_path)
        logger.debug("Data ingestion process completed successfully")

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
