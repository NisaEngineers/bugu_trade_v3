import requests
import json
from transformers import pipeline
import warnings

# Ignore only DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

class NewsSentimentAnalyzer:
    def __init__(self, api_key, symbol='BTCUSD', language='en', page_size=5):
        self.api_key = api_key
        self.symbol = symbol
        self.language = language
        self.page_size = page_size
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert")

    def fetch_latest_news(self):
        url = f'https://newsapi.org/v2/everything?q={self.symbol}&language={self.language}&pageSize={self.page_size}&apiKey={self.api_key}'
        response = requests.get(url)
        return response.json()

    def analyze_sentiment(self, text):
        if text:  # Check if the text is not None or empty
            return self.pipe(text)[0]
        else:
            return {"label": "neutral", "score": 0.0}

    def run_analysis(self):
        news_data = self.fetch_latest_news()
        if news_data['status'] == 'ok':
            total_score = 0
            num_article = 0

            for article in news_data['articles']:
                description = article['description']
                sentiment = self.analyze_sentiment(description)
                print(f"Title: {article['title']} Sentiment: {sentiment['label']}, Score: {sentiment['score']}")

                if sentiment["label"] == "positive":
                    total_score += sentiment["score"]
                elif sentiment["label"] == "negative":
                    total_score -= sentiment["score"]

                num_article += 1

            if num_article > 0:
                final = total_score / num_article
                print(f'Overall sentiment: {"Positive" if final > 0 else "Negative" if final < 0 else "Neutral"} ({final})')
            else:
                print('No articles found.')
        else:
            print("Failed to fetch news")

# Usage:
# api_key = '5cb98f18e3224b98bc1ddef0446b4dc0'  # Replace 'your_api_key' with your actual NewsAPI key
# symbol = input("Enter the symbol (e.g., BTCUSD): ")
# page_size = int(input("Enter the number of latest news articles to fetch: "))

# news_analyzer = NewsSentimentAnalyzer(api_key, symbol=symbol, page_size=page_size)
# news_analyzer.run_analysis()
