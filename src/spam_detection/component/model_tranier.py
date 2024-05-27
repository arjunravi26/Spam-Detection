from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.spam_detection.logging import logging
import pandas as pd
from src.spam_detection.utils.common import save_model
class ModelTrainer:
    def __init__(self,config) -> None:
         self.config = config 
    def split(self,data):
        self.X = data.drop('output',axis=1)
        self.y = data['output']
        
    def model_train(self):
        # Gradient Boosting
        self.model = SVC(kernel='linear')
        self.model.fit(self.X,self.y )
        
    def evaluate(self,data):
        X_test = data.drop('output',axis=1)
        y_test = data['output']
        pred = self.model.predict(X_test)
        return accuracy_score(pred,y_test)
    
    def train(self,train_data_path,eval_data_path):
        train_data = pd.read_csv(train_data_path)
        eval_data = pd.read_csv(eval_data_path)
        self.split(train_data)
        self.model_train()
        score = self.evaluate(eval_data)
        logging.warning(f'Gradient Boost have {score} score')
        # self.save_model()
    def final_train(self,df):
        self.split(df)
        self.model_train()
        save_model(self.model,self.config.model_path)
        
        
    
    