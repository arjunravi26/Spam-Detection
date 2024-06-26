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
    # def calculate_tf_idf(self,data):
    #     documents = [' '.join(doc) for doc in data]
    #     # Initialize TF-IDF Vectorizer
    #     self.tfidf_vectorizer = TfidfVectorizer()
    #     self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    #     self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
    #     # Convert TF-IDF matrix to DataFrame
    #     self.tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.feature_names)
    #     with open(self.config.tfidf_path, 'wb') as file:
    #         pickle.dump(file=file,obj=self.tfidf_vectorizer)
            
    # def get_weighted_word2vec(self,doc, doc_idx):
    #     # words = doc.split()
    #     tfidf_scores = {word: self.tfidf_matrix[doc_idx, self.tfidf_vectorizer.vocabulary_[word]] for word in doc if word in self.tfidf_vectorizer.vocabulary_}
    #     weighted_vectors = []

    #     for word in doc:
    #         if word in self.model.index_to_key:
    #             word_vector = self.model[word]
    #             tfidf_weight = tfidf_scores.get(word, 0)
    #             weighted_vectors.append(tfidf_weight * word_vector)

    #     return np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(self.model.vector_size)
    
    # def test_get_weighted_word2vec(self,doc, tfidf_vectorizer, tfidf_matrix):
    #     words = doc.split()
    #     print(words)
    #     tfidf_scores = {word: tfidf_matrix[0, tfidf_vectorizer.vocabulary_[word]] for word in words if word in tfidf_vectorizer.vocabulary_}
    #     weighted_vectors = []
    #     for word in doc:
    #         if word in self.model.index_to_key:
    #             word_vector = self.model[word]
    #             tfidf_weight = tfidf_scores.get(word, 0)
    #             weighted_vectors.append(tfidf_weight * word_vector)

    #     return np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(self.model.vector_size)




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
            # self.calculate_tf_idf(train_words+test_words+validation_words)
            train_embeddings = [self.avg_word2vec(doc) for doc in tqdm(train_words,desc="Transforming traning words into average word2vec")]
            test_embeddings = [self.avg_word2vec(doc) for doc in tqdm(test_words,desc="Transforming testing words into average word2vec")]
            validation_embeddings = [self.avg_word2vec(doc) for doc in tqdm(validation_words,desc="Transforming validation words into average word2vec")]

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
            
            corpus = [self.preprocess_text(message) for message in tqdm(data_df['message'],desc="Preprocessing message")]
            words = [simple_preprocess(sent) for doc in corpus for sent in tqdm(sent_tokenize(doc),desc="Sent tokenizing")]

            # with open(self.config.tfidf_path, 'rb') as f:
            #     tfidf_vectorizer = pickle.load(f)
            # documents = ' '.join(words[0]) 
            transformed_data = [self.avg_word2vec(words[0])]
            
            return np.array(transformed_data)
        except Exception as e:
            logging.error(f"Error in preprocessing new data: {e}")
            raise e
