from flask import Flask, render_template, request
from main import SpamDetector
app = Flask(__name__)
spam_detector = SpamDetector()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_spam', methods=['POST'])
def check_spam():
    msg = request.form['message']
    try:
        output = spam_detector.detect(msg)
        # Determine the result based on the output
        if output == True:
            result = "Not Spam"
        else:
            result = "Spam"
        return f"Your Message '{msg}' is {result}"
    
    except Exception as e:
        return e


if __name__ == '__main__':
    app.run(debug=False)
