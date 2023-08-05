from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained sentiment analysis model and data
sid = joblib.load('sentiment_model.joblib')
data = pd.read_csv('sentiment_analysis_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    if text.strip():
        sentiment, sentiment_score = analyze_sentiment(text)
    else:
        sentiment, sentiment_score = "N/A", 0.0
    return render_template('result.html', text=text, sentiment=sentiment, sentiment_score=sentiment_score)

def analyze_sentiment(text):
    if isinstance(text, str):
        scores = sid.polarity_scores(text)
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        return sentiment, scores['compound']
    else:
        return "N/A", 0.0

if __name__ == '__main__':
    app.run(debug=True)
