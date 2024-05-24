from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
import pickle
from src.spam_detection.config.configuration import ConfiguaraionManager


class Prediction:
    def __init__(self) -> None:
        config = ConfiguaraionManager()
        model_path_config = config.model_trainer()
        self.model = pickle.load(model_path_config.model_path)
        self.preprocess_pipeline = DataTransformationPipeline()

    def preprocess(self, data):
        """Preprocesses the data using the DataTransformationPipeline."""
        if data is None:
            raise ValueError("Data cannot be None.")
        preprocessed_data = self.preprocess_pipeline.predict(data)
        return preprocessed_data

    def model_predict(self, predict_data):
        """Makes a prediction using the preloaded model."""
        if predict_data is None:
            raise ValueError("Predict data cannot be None.")
        output = self.model.predict(predict_data)
        return output

    def start(self, data):
        predict_data = self.preprocess(data)
        output = self.model_predict(predict_data)
        return output
