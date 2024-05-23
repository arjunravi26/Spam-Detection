from src.spam_detection.component.data_ingestion import DataIngestion
from src.spam_detection.config.configuration import ConfiguaraionManager

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass
    def start(self):
        config = ConfiguaraionManager()
        data_ingestion_config = config.data_ingestion()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.data_ingestion()
        