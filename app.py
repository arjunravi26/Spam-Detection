from flask import Flask, render_template, request
from main import SpamDetector
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_spam', methods=['POST'])
def check_spam():
    msg = request.form['message']
    if not msg:
        return "Message is empty"
    else:
        spam_dectector = SpamDetector()
        result = spam_dectector.predict(msg)

    print("Received message:", msg)
    return "The message " + msg + " is " + result

if __name__ == '__main__':
    app.run(debug=True)
