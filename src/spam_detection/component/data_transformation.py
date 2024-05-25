import re
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
from tqdm import tqdm
from src.spam_detection.entity import DataTransformationConfig
from src.spam_detection.config.configuration import ConfiguaraionManager
from src.spam_detection.logging import logging
from src.spam_detection.utils.common import save_data

class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = self.load_word2vec_model(config.model_path)
        
    def load_word2vec_model(self, path: str) -> KeyedVectors:
        try:
            print("Loading Word2Vec model...")
            word2vec_model = KeyedVectors.load(path)
            print("Word2Vec model loaded.")
            return word2vec_model
        except Exception as e:
            logging.error(f"Error loading Word2Vec model: {e}")
            raise e
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub('[^A-Za-z]', ' ', text)
        text = text.lower().split()
        text = [self.lemmatizer.lemmatize(word) for word in text]
        return ' '.join(text)
    
    def transform_dataset(self, data: pd.DataFrame):
        corpus = [self.preprocess_text(message) for message in data['message']]
        data = data[corpus]
        y = pd.get_dummies(data['label'], drop_first=True).values.ravel()
        words = [simple_preprocess(sent) for doc in corpus for sent in sent_tokenize(doc)]
        return words, y

    def avg_word2vec(self, doc):
        vectors = [self.word2vec_model.wv[word] for word in doc if word in self.word2vec_model.wv.key_to_index]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word2vec_model.vector_size)
    
    def preprocess_data(self):
        try:
            config = ConfiguaraionManager()
            data_ingestion_config = config.data_ingestion()
            
            train_data = pd.read_csv(data_ingestion_config.train_path)
            test_data = pd.read_csv(data_ingestion_config.test_path)
            validation_data = pd.read_csv(data_ingestion_config.validation_path)
    
            train_words, train_y = self.transform_dataset(train_data)
            test_words, test_y = self.transform_dataset(test_data)
            validation_words, validation_y = self.transform_dataset(validation_data)
        
            X_train = [self.avg_word2vec(doc) for doc in tqdm(train_words, desc="Processing training data")]
            X_test = [self.avg_word2vec(doc) for doc in tqdm(test_words, desc="Processing test data")]
            X_val = [self.avg_word2vec(doc) for doc in tqdm(validation_words, desc="Processing validation data")]
            
            save_data(self.config.train_path, X_train, train_y)
            save_data(self.config.test_path, X_test, test_y)
            save_data(self.config.validation_path, X_val, validation_y)
        
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise e

    def preprocess_new_data(self, data):
        try:
            if isinstance(data, str):
                data = [data]
            if isinstance(data, list):
                data = pd.Series(data)
            data_df = pd.DataFrame({'message': data})
            
            corpus = [self.preprocess_text(message) for message in data_df['message']]
            words = [simple_preprocess(sent) for doc in corpus for sent in sent_tokenize(doc)]
            transformed_data = [self.avg_word2vec(doc) for doc in words]
            
            return np.array(transformed_data)
        except Exception as e:
            logging.error(f"Error in preprocessing new data: {e}")
            raise e
