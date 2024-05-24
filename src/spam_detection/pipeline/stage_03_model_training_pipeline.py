from src.spam_detection.config.configuration import ConfiguaraionManager
from src.spam_detection.component.model_tranier import ModelTrainer
import pandas as pd
from src.spam_detection.logging import logging
class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    def start(self):
        config = ConfiguaraionManager()
        model_training_config = config.model_trainer()
        model_training = ModelTrainer(config=model_training_config)
        data_config = config.data_transformation()
        model_training.train(train_data_path=data_config.train_path, eval_data_path=data_config.validation_path)
        df1 = pd.read_csv(data_config.train_path)
        df2 =  pd.read_csv(data_config.test_path)
        df3 =  pd.read_csv(data_config.validation_path)
        df = pd.concat([df1, df2, df3])
        model_training.final_train(df)
        logging.info("Model Saved")


