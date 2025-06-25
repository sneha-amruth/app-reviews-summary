import pandas as pd
import datetime
from google_play_scraper import reviews, Sort
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_data
def load_reviews_for_app(app_name, app_id):
    """
    Fetches all reviews for a single app from October 1, 2024 to the present.
    """
    start_date = datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)
    
    all_app_reviews = []
    continuation_token = None
    max_pages = 20  # Safety break
    pages_fetched = 0
    
    while pages_fetched < max_pages:
        try:
            result, token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token
            )
            
            if not result:
                break

            continuation_token = token
            pages_fetched += 1
            
            new_reviews = [r for r in result if r['at'].replace(tzinfo=datetime.timezone.utc) >= start_date]
            all_app_reviews.extend(new_reviews)

            if len(new_reviews) < len(result):
                break # Found reviews older than the start date
            
            if not token:
                break # No more pages

        except Exception as e:
            st.error(f"An error occurred while fetching reviews for {app_name}: {e}")
            break
    
    if not all_app_reviews:
        return pd.DataFrame()

    df = pd.DataFrame(all_app_reviews)
    df['app'] = app_name
    df = df.rename(columns={'content': 'review', 'at': 'date'})
    return df[['app', 'review', 'date', 'score']].sort_values(by='date', ascending=False).reset_index(drop=True)

def analyze_reviews_with_vader(df, progress_bar=None, status_text=None):
    """
    Analyzes reviews using the local VADER model for fast, reliable sentiment analysis.
    """
    if df.empty:
        return df, "No reviews to analyze."

    app_name = df['app'].iloc[0]
    analyzer = SentimentIntensityAnalyzer()
    all_sentiments = []
    total_reviews = len(df)

    if status_text:
        status_text.text(f"[{app_name}] Analyzing {total_reviews} reviews with VADER...")

    def get_sentiment_from_vader(compound_score):
        if compound_score >= 0.05:
            return "Positive"
        if compound_score <= -0.05:
            return "Negative"
        return "Neutral"

    for i, row in df.iterrows():
        review_text = str(row['review'])
        vader_scores = analyzer.polarity_scores(review_text)
        sentiment = get_sentiment_from_vader(vader_scores['compound'])
        all_sentiments.append(sentiment)
        
        if progress_bar:
            progress_bar.progress((i + 1) / total_reviews)

    df['sentiment'] = all_sentiments

    # Calculate sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    total_sentiments = len(df['sentiment'])

    positive_perc = (sentiment_counts.get('Positive', 0) / total_sentiments) * 100 if total_sentiments > 0 else 0
    negative_perc = (sentiment_counts.get('Negative', 0) / total_sentiments) * 100 if total_sentiments > 0 else 0
    neutral_perc = (sentiment_counts.get('Neutral', 0) / total_sentiments) * 100 if total_sentiments > 0 else 0

    summary = (
        f"""**Sentiment Distribution:**  

        - **Positive:** {positive_perc:.1f}% ({sentiment_counts.get('Positive', 0)})  

        - **Negative:** {negative_perc:.1f}% ({sentiment_counts.get('Negative', 0)})  

        - **Neutral:** {neutral_perc:.1f}% ({sentiment_counts.get('Neutral', 0)})"""
    )
    
    return df, summary
