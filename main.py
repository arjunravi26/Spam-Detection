from src.spam_detection.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.spam_detection.pipeline.stage_03_model_training_pipeline import ModelTrainingPipeline
from src.spam_detection.pipeline.stage_04_prediction import PredictionPipeline
import pandas as pd
import warnings
from datetime import datetime

class SpamDetector:
    def __init__(self,model)->None:
        self.model = model
        pass
    def train(self)->None:
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.start()
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.start()
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.start()
    def detect(self, message):
        predict_pipeline = PredictionPipeline(message,self.model)
        start_time = datetime.now()
        print(start_time)
        output = predict_pipeline.start()
        
        # Determine the result based on the output
        if output == True:
            result = "Not Spam"
        else:
            result = "Spam"
        
        end_time = datetime.now()
        print(end_time)
        time_taken = end_time - start_time
        print(f"Time taken is {time_taken}")
        return result


# if __name__ == "__main__":
#     spam_dectector = SpamDetector()
#     msg = input("Enter a message to check if it is spam or not: ")
#     if not msg:
#         msg = "Special offer just for you! Get 50% off on all products. Visit our website now!"
#     print(spam_dectector.predict(message=msg))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#     message_list = [
#     "Get a loan approved within minutes. No credit check required. Apply now!",
#     "Congratulations! You've won a free iPhone. Click the link to claim your prize.",
#     "Your account has been compromised. Click here to verify your identity immediately.",
#     "You've been selected as a winner of our $1,000,000 lottery! Contact us to claim your prize.",
#     "Special offer just for you! Get 50% off on all products. Visit our website now!",
#     "Special offer just for you! Get 50% off on all products. Visit our website now!",
#     "Earn $5000 per week from the comfort of your home. No experience needed. Sign up today!",
#     "Your PayPal account has been suspended. Please click here to update your information."
# ]



    # result_list = []
    # for message in message_list:
    #     prediction_pipeline = PredictionPipeline(message)
    #     output = prediction_pipeline.start()
        
    #     if output == True:
    #         result = "Not Spam"
    #     else:
    #         result = "Spam"
    #     result_list.append(result)
    # print(result_list)
        
    