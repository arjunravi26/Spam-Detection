from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
import pandas as pd
from src.spam_detection.utils.common import *

class Prediction:
    def __init__(self,model) -> None:
        self.model = model
        self.preprocess_pipeline = DataTransformationPipeline()

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the data using the DataTransformationPipeline."""
        if data is None:
            raise ValueError("Input data cannot be None.")
        preprocessed_data = self.preprocess_pipeline.predict_transform(data)
        return preprocessed_data

    def model_predict(self, predict_data: pd.DataFrame) -> pd.Series:
        """Makes a prediction using the preloaded model."""
        if predict_data is None:
            raise ValueError("Predict data cannot be None.")
        output = self.model.predict(predict_data)
        return output

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Main method to preprocess data and make predictions."""
        predict_data = self.preprocess(data)
        output = self.model_predict(predict_data)
        return output
