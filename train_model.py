import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

# Download VADER lexicon (only needed for the first time)
nltk.download('vader_lexicon')

# Load the dataset
data = pd.read_csv("amazon_cells_review.csv")  # Replace "Reviews.csv" with the actual filename of the downloaded dataset

# Create a SentimentIntensityAnalyzer object
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment using VADER and return sentiment label and score
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

# Add a new column 'Sentiment' to the dataset with sentiment labels
data['Sentiment'], data['Sentiment_Score'] = zip(*data['body'].apply(analyze_sentiment))

# Save the trained sentiment analysis model and data to use in the Flask application
joblib.dump(sid, 'sentiment_model.joblib')
data.to_csv('sentiment_analysis_data.csv', index=False)

print("Sentiment analysis model trained and data saved.")
