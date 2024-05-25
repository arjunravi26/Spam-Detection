from flask import Flask, render_template, request
from main import SpamDetector
from gensim.models import KeyedVectors
from src.spam_detection.config.configuration import ConfiguaraionManager
from datetime import datetime

app = Flask(__name__)

# Global variables for model and spam detector
word2vec_model = None
spam_detector = None

def load_model():
    global word2vec_model, spam_detector
    if word2vec_model is None:
        start_time = datetime.now()
        print(start_time)
        print("Loading Word2Vec model...")
        config = ConfiguaraionManager()
        model_path_config = config.data_transformation()
        model_path = model_path_config.model_path
        word2vec_model = KeyedVectors.load(model_path)
        spam_detector = SpamDetector(word2vec_model)
        end_time = datetime.now()
        print("Word2Vec model loaded.")
        print(end_time)
        print(f"Time taken {end_time-start_time}")

# Ensure the model is loaded when the script is executed directly
load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_spam', methods=['POST'])
def check_spam():
    msg = request.form['message']
    if not msg:
        return "Message is empty"
    try:
        result = spam_detector.detect(msg)
        print("Received message:", msg, " is ", result)
        return f"Your Message '{msg}' is {result}"
    except Exception as e:
        print(f"Error processing message: {e}")
        return "An error occurred while processing your message."

if __name__ == '__main__':
    app.run(debug=False)
