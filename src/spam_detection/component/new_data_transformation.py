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
            
    
    def test_get_weighted_word2vec(self,doc, tfidf_vectorizer, tfidf_matrix):
        print(doc)
        tfidf_scores = {word: tfidf_matrix[0, tfidf_vectorizer.vocabulary_[word]] for word in doc if word in tfidf_vectorizer.vocabulary_}
        weighted_vectors = []
        for word in doc:
            if word in self.model.index_to_key:
                word_vector = self.model[word]
                tfidf_weight = tfidf_scores.get(word, 0)
                weighted_vectors.append(tfidf_weight * word_vector)

        return np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(self.model.vector_size)

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

            transformed_data = [self.test_get_weighted_word2vec(doc) for doc in words]
            
            return np.array(transformed_data)
        except Exception as e:
            logging.error(f"Error in preprocessing new data: {e}")
            raise e
