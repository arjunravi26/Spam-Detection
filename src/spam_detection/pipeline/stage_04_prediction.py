from src.spam_detection.component.prediction import Prediction

class PredictionPipeline:
    def __init__(self,data) -> None:
        self.data = data

    def start(self):
        prediction_pipeline = Prediction()
        output = prediction_pipeline.predict(data=self.data)
        return output