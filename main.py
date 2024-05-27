from src.spam_detection.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.spam_detection.pipeline.stage_03_model_training_pipeline import ModelTrainingPipeline
from src.spam_detection.pipeline.stage_04_prediction import PredictionPipeline
from src.spam_detection.utils.load_models import load_word2vec_model
from datetime import datetime
import warnings

class SpamDetector:
    def __init__(self,model)->None:
        self.model = model
        
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
        
        # # Determine the result based on the output
        # if output == True:
        #     result = "Not Spam"
        # else:
        #     result = "Spam"
        
        end_time = datetime.now()
        print(end_time)
        time_taken = end_time - start_time
        print(f"Time taken is {time_taken}")
        return output


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    word2vec_model = load_word2vec_model()
    spam_dectector = SpamDetector(model=word2vec_model)
    spam_dectector.train()
    # msg = input("Enter a message to check if it is spam or not: ")
    # if not msg:
    #     msg = "Special offer just for you! Get 50% off on all products. Visit our website now!"
    # print(spam_dectector.detect(message=msg))
    

    
# if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    message_list = [
    "Get a loan approved within minutes. No credit check required. Apply now!",
    "Congratulations! You've won a free iPhone. Click the link to claim your prize.",
    "Your account has been compromised. Click here to verify your identity immediately.",
    "You've been selected as a winner of our $1,000,000 lottery! Contact us to claim your prize.",
    "Special offer just for you! Get 50% off on all products. Visit our website now!",
    "Earn $5000 per week from the comfort of your home. No experience needed. Sign up today!",
    "Your PayPal account has been suspended. Please click here to update your information.",
    "Get a free trial of our premium service. No credit card required. Sign up now!",
    "YOU WON! FREE PHONE! Just click here [gibberish_link.com] to claim yours! **",
    "URGENT: Your account will be closed if you don't verify your information immediately.",
    "Make $10,000 in just 1 week! No experience needed. Click here to learn more.",
    "Earn $$$ from home! No experience required! Click here for more info!",
    "Get a free iPhone! Just click here [gibberish_link.com] to claim yours! **",
    "URGENT: Your account has been compromised. Click here to verify your identity immediately.",
    "Tired of your dead-end job? Change your life with our amazing product! Ask me how!",
    "Attached: Invoice #12345. Payment due immediately.",
    "Double your money in weeks with our exclusive crypto scheme! Limited spots available!",
    "Hi [Your Name], Reminder! Movie night at my place tonight at 7pm. Can't wait to see you there!",
    "Instagram update: You have 2 new friend requests!",
    "Hi Arjun, I'm so sorry for missing your birthday. Let's catch up soon!",
    "We're excited to share our latest company newsletter with you! In this issue, you'll find information about our new product launch, upcoming events, and more. Click here to read the full newsletter."
]

    result_list = []
    # word2vec_model = load_word2vec_model()
    # spam_detector = SpamDetector(word2vec_model)

    for message in message_list:
        output = spam_dectector.detect(message=message)
        result_list.append(output)
    print(result_list)
        
    