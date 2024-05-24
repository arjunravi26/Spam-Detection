from src.spam_detection.config.configuration import ConfiguaraionManager
from src.spam_detection.component.model_tranier import ModelTrainer
class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    def start(self):
        config = ConfiguaraionManager()
        model_training_config = config.model_trainer()
        model_training = ModelTrainer(config=model_training_config)
        data_config = config.data_transformation()
        model_training.train(train_data_path=data_config.train_path, eval_data_path=data_config.validation_path)
