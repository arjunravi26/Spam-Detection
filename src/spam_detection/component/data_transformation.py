import re
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from src.spam_detection.entity import DataTransformationConfig
from src.spam_detection.config.configuration import ConfiguaraionManager
from src.spam_detection.logging import logging
from src.spam_detection.utils.common import save_data
import pickle

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, model: KeyedVectors) -> None:
        self.config = config
        self.model = model
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer()

    def preprocess_text(self, text: str) -> str:
        text = re.sub('[^A-Za-z]', ' ', text)
        text = text.lower().split()
        text = [self.lemmatizer.lemmatize(word) for word in text]
        return ' '.join(text)

    def transform_dataset(self, data: pd.DataFrame):
        preprocessed_corpus = [self.preprocess_text(message) for message in data['message']]
        data = data[list(map(lambda x: len(x) > 0, preprocessed_corpus))]
        y = pd.get_dummies(data['label'])
        y = y.iloc[:, 0].values
        words = [simple_preprocess(sent) for doc in preprocessed_corpus for sent in sent_tokenize(doc)]
        return words, y
    
    def avg_word2vec(self, doc):
        vectors = [self.model[word] for word in doc if word in self.model.key_to_index]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

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

            train_embeddings = [self.avg_word2vec(doc) for doc in train_words]
            test_embeddings = [self.avg_word2vec(doc) for doc in test_words]
            validation_embeddings = [self.avg_word2vec(doc) for doc in validation_words]

            train_embeddings = np.array(train_embeddings)
            test_embeddings = np.array(test_embeddings)
            validation_embeddings = np.array(validation_embeddings)

            save_data(self.config.train_path, train_embeddings, train_y)
            save_data(self.config.test_path, test_embeddings, test_y)
            save_data(self.config.validation_path, validation_embeddings, validation_y)
        
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

            with open(self.config.tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)

            transformed_data = [self.avg_word2vec(doc) for doc in words]
            
            return np.array(transformed_data)
        except Exception as e:
            logging.error(f"Error in preprocessing new data: {e}")
            raise e
