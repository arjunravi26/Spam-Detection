from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from src.spam_detection.logging import logging
import pickle
import pandas as pd
from src.spam_detection.utils.common import *
class ModelTrainer:
    def __init__(self,config) -> None:
         self.config = config 
    def split(self,data):
        self.X = data.drop('output',axis=1)
        self.y = data['output']
        
    def model_train(self):
        # Gradient Boosting
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X,self.y )
        
    def evaluate(self,data):
        X_test = data.drop('output',axis=1)
        y_test = data['output']
        pred = self.model.predict(X_test)
        return accuracy_score(pred,y_test)
    
    def save_model(self):
        with open(self.config.model_path, 'wb') as file:
            pickle.dump(self.model, file)
            logging.info(f"Model saved to {file}")
    
    def train(self,train_data_path,eval_data_path):
        train_data = pd.read_csv(train_data_path)
        eval_data = pd.read_csv(eval_data_path)
        self.split(train_data)
        self.model_train()
        score = self.evaluate(eval_data)
        logging.warning(f'Random Forest have {score} score')
        # self.save_model()
    def final_train(self,df):
        self.split(df)
        self.model_train()
        save_model(self.model,self.config.model_path)
        
        
    
    