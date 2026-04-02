from newsapi import NewsApiClient
from textblob import TextBlob

NEWS_API_KEY = "b79e450092534ba1b6c720a910466a77"

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def get_news_sentiment(query):
    try:
        articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )

        sentiments = []

        for article in articles["articles"]:
            title = article["title"]

            polarity = TextBlob(title).sentiment.polarity
            sentiments.append(polarity)

        if len(sentiments) == 0:
            return 0

        return sum(sentiments) / len(sentiments)

    except Exception as e:
        print("Error:", e)
        return 0