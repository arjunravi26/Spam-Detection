from sklearn.ensemble import RandomForestClassifier
from src.spam_detection.config.configuration import ConfiguaraionManager
import pandas as pd
class ModelTrainer:
    def __init__(self) -> None:
         pass 
    def train(self):
        config = ConfiguaraionManager()
        config_data_path = config.data_ingestion()
        message = pd.read_csv(config_data_path.train_path)
        y = message[list(map(lambda x: len(x)>0))]