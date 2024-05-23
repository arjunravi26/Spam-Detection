from src.spam_detection.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.start()