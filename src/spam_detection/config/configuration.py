from src.spam_detection.entity import DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig
from src.spam_detection.utils.common import *
from src.spam_detection.constant import *


class ConfiguaraionManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file=PARAMS_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_file_path)
        self.parmas = read_yaml(params_file)
        create_directory([self.config.artifacts_root])

    def data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directory([config.root_dir])
        self.data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir, data_path=config.data_path,train_path = config.train_path,
            test_path = config.test_path, validation_path = config.validation_path
        )
        return self.data_ingestion_config
    
    def data_transformation(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directory([config.root_dir])
        self.data_ingestion_config = DataTransformationConfig(
            root_dir=config.root_dir,train_path = config.train_path,
            test_path = config.test_path, validation_path = config.validation_path,
            model_path = config.model_path
            
        )
        return self.data_ingestion_config
    def model_trainer(self) -> ModelTrainerConfig:
        config = self.config.model_tranier
        create_directory([config.root_dir])
        self.data_ingestion_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_path = config.model_path
            
        )
        return self.data_ingestion_config

