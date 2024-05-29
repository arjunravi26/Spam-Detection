from src.spam_detection.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.spam_detection.pipeline.stage_03_model_training_pipeline import ModelTrainingPipeline
from src.spam_detection.pipeline.stage_04_prediction import PredictionPipeline
from src.spam_detection.utils.load_models import load_word2vec_model
from datetime import datetime
import warnings

class SpamDetector:
    def __init__(self)->None:
        self.model = load_word2vec_model()
        
    def train(self)->None:
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.start()
        data_transformation_pipeline = DataTransformationPipeline(self.model)
        data_transformation_pipeline.start()
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.start()
    def detect(self, message):
        warnings.filterwarnings("ignore")
        predict_pipeline = PredictionPipeline(message,self.model)
        start_time = datetime.now()
        print(start_time)
        output = predict_pipeline.start()
        
        # Determine the result based on the output
        # if output == True:
        #     result = "Not Spam"
        # else:
        #     result = "Spam"
        
        end_time = datetime.now()
        print(end_time)
        time_taken = end_time - start_time
        print(f"Time taken is {time_taken}")
        return output