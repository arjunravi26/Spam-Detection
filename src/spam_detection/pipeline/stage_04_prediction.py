from src.spam_detection.component.prediction import Prediction

class PredictionPipeline:
    def __init__(self,data,model) -> None:
        self.data = data
        self.model = model

    def start(self):
        prediction_pipeline = Prediction(model=self.model)
        output = prediction_pipeline.predict(data=self.data)
        return output