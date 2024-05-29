from sklearn.metrics import accuracy_score,confusion_matrix
from main import SpamDetector
from src.spam_detection.config.configuration import ConfiguaraionManager
import warnings
import pandas as pd
if __name__ == "__main__":
    config = ConfiguaraionManager()
    test_path_config = config.model_tester()
    test_path = test_path_config.test_path
    df = pd.read_csv(test_path)
    X = df['messages']
    y = df['results']
    warnings.filterwarnings("ignore")
    spam_dectector = SpamDetector()

    result_list = []
    for message in X:
        output = spam_dectector.detect(message=message)
        result_list.append(output)
    print(result_list)
   
    print(f"Total no of Spam and Non spam msg are: {y.value_counts()}")
    print(f"Accuracy: {accuracy_score(result_list,y)}")
    print(f"Confussion Matrix:\n{confusion_matrix(result_list,y)}")
    
    

    # Compare the predicted results with the true results
    incorrectly_classified_indices = [i for i, (predicted, true) in enumerate(zip(y, result_list)) if predicted != true]

    # Print the messages that were incorrectly classified
    for index in incorrectly_classified_indices:
        print(f"Message: {X[index]}")
        print(f"True label: {y[index]}")
        print(f"Predicted label: {result_list[index]}\n")

        
    