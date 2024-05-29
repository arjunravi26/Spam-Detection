from train_test_model.train import SpamDetector
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    spam_dectector = SpamDetector()
    ## Uncomment if you want to retrain the model
    # spam_dectector.train()
    msg = input("Enter a message to check if it is spam or not: ")
    if msg:
        print(spam_dectector.detect(message=msg))