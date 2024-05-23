from src.spam_detection.entity import DataIngestionConfig
from src.spam_detection.logging import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import os


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def data_ingestion(self):
        base_dir = "Data"
        file_name = "SMSSpamCollection.txt"
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            try:
                message = pd.read_csv(file_path, names=["label", "message"], sep="\t")
                message.to_csv(self.config.data_path, index=False)
                train_df, temp_df = train_test_split(message, test_size=0.4, random_state=42)
                validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
                train_df.to_csv(self.config.train_path, index=False)
                test_df.to_csv(self.config.test_path, index=False)
                validation_df.to_csv(self.config.validation_path, index=False)
                logging.info(f"Data ingestion completed successfully. Data saved to {self.config.data_path}.")
            except Exception as e:
                logging.error(f"Error during data ingestion: {e}")
                raise e
        else:
            logging.error(f"Error in data ingestion: path {file_path} does not exist")
            raise FileNotFoundError(f"Path {file_path} does not exist")
