from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
import pickle
from src.spam_detection.config.configuration import ConfiguaraionManager
class Prediction:
    def __init__(self) -> None:
        pass
    def preprocess(self,data):
        preprocess_pipeline = DataTransformationPipeline()
        preproecessed_data = preprocess_pipeline.predict(data)
        return preproecessed_data
    def model_predict(self,predict_data):
        config = ConfiguaraionManager()
        model_path_config = config.model_trainer()
        model = pickle.load(model_path_config.model_path)
        output = model.predict(predict_data)
        return output
    def start(self,data):
        predict_data = self.preprocess(data)
        output = self.model_predict(predict_data)
        return output