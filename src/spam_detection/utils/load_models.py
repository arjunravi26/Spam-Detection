from gensim.models import KeyedVectors
from src.spam_detection.config.configuration import ConfiguaraionManager
from src.spam_detection.logging import logging
import pickle
from ensure import ensure_annotations
from datetime import datetime

def load_word2vec_model() -> KeyedVectors:
        try:
            start_time = datetime.now()
            print(f"Loading Word2Vec model...time {start_time}")
            config = ConfiguaraionManager()
            model_path_config = config.data_transformation()
            word2vec_model = KeyedVectors.load(model_path_config.model_path)
            end_time = datetime.now()
            print(f"Word2Vec model loaded. at {end_time}\nTime taken is {end_time-start_time} ")
            return word2vec_model
        except Exception as e:
            logging.error(f"Error loading Word2Vec model: {e}")
            raise e
        
@ensure_annotations
def load_model():
    try:
        config = ConfiguaraionManager()
        model_config = config.model_trainer()
        path = model_config.model_path
        print("Model loading started")
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded")
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        raise e