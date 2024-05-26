from flask import Flask, render_template, request
from main import SpamDetector
from src.spam_detection.utils.load_models import load_word2vec_model
app = Flask(__name__)

# Global variables for model and spam detector
word2vec_model = None
spam_detector = None

def load_model():
    global word2vec_model, spam_detector
    if word2vec_model is None:
        word2vec_model = load_word2vec_model()
        spam_detector = SpamDetector(word2vec_model)    

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
        print(f"Received message: {msg} is {result}")
        return f"Your Message '{msg}' is {result}"
    except Exception as e:
        print(f"Error processing message: {e}")
        return "An error occurred while processing your message."

if __name__ == '__main__':
    app.run(debug=False)
