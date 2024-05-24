from src.spam_detection.entity import DataTransformationConfig
from nltk.stem import WordNetLemmatizer
from src.spam_detection.config.configuration import ConfiguaraionManager
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.spam_detection.logging import logging
import gensim.downloader as api
from src.spam_detection.utils.common import *

def avg_word2vec(doc,model):
    vectors = [model.wv[word] for word in doc if word in model.wv.index_to_key]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
class DataTransformation:
    def __init__(self,config:DataTransformationConfig) -> None:
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = None
        
    def preprocess_text(self,text):
        review = re.sub('[^A-Za-z]',' ',text)
        review = review.lower()
        review = review.split()
        words = [self.lemmatizer.lemmatize(word) for word in review]
        words = ' '.join(words)
        return words
    
    def transform_dataset(self,data):
        corpus = [self.preprocess_text(message) for message in data['message']]
        y = data[list(map(lambda x: len(x)>0,corpus))]
        y = pd.get_dummies(y['label'])
        y = y.iloc[:,0].values
        words = [simple_preprocess(sent) for doc in corpus for sent in sent_tokenize(doc)]
        return words,y

    def preprocess_data(self):
        try:
            config = ConfiguaraionManager()
            data_ingestion_config = config.data_ingestion()
            train_data = pd.read_csv(data_ingestion_config.train_path)
            test_data = pd.read_csv(data_ingestion_config.test_path)
            validation_data = pd.read_csv(data_ingestion_config.validation_path)
    
            
            train_words,train_y = self.transform_dataset(train_data)
            test_words,test_y = self.transform_dataset(test_data)
            validation_words,validation_y = self.transform_dataset(validation_data)
        
            
            # self.word2vec_model = Word2Vec(sentences=train_words,vector_size=100,window=5,min_count=1)
            # self.word2vec_model = api.load('word2vec-google-news-300')
            # Load Google's pre-trained Word2Vec model
            #self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

            if self.word2vec_model is None:
                self.word2vec_model = load_model(self.config.model_path)
            X_train = [avg_word2vec(doc,self.word2vec_model) for doc in tqdm(train_words)]
            X_test = [avg_word2vec(doc,self.word2vec_model) for doc in tqdm(test_words)]
            X_val = [avg_word2vec(doc,self.word2vec_model) for doc in tqdm(validation_words)]
            
            save_data(self.config.train_path,X_train,train_y)
            save_data(self.config.test_path,X_test,test_y)
            save_data(self.config.validation_path,X_val,validation_y)
        
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise e
    def preprocess_new_data(self,data):
        
        try:
            if isinstance(data, str):
                new_data = [data]
        
            if isinstance(data, list):
                new_data = pd.Series(data)
            if self.word2vec_model is None:
                self.word2vec_model = load_model(self.config.model_path)
            data = pd.DataFrame({'message': new_data})
            corpus = [self.preprocess_text(message) for message in data['message']]
            words = [simple_preprocess(sent) for doc in corpus for sent in sent_tokenize(doc)]

            
            transformed_data = [avg_word2vec(doc, self.word2vec_model) for doc in words]
            return np.array(transformed_data)
        except Exception as e:
            raise e
                    
