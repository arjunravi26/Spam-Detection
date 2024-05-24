from src.spam_detection.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.spam_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.spam_detection.pipeline.stage_03_model_training_pipeline import ModelTrainingPipeline



if __name__ == "__main__":
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.start()
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.start()
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.start()
    
    