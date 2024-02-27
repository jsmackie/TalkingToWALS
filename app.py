from flask import Flask, render_template, request
import HidingBehindWALS as wals

app = Flask(__name__)

agent = wals.WALS()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_text():
    text = request.form['text']
    response = agent.TalkToWALS(text)
    return response

if __name__ == '__main__':
    app.run(debug=True)