import pandas as pd
import datetime
from google_play_scraper import reviews, Sort
import streamlit as st
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_resource
def get_vader_analyzer():
    """Initializes and caches the VADER sentiment analyzer."""
    return SentimentIntensityAnalyzer()

@st.cache_data
def load_reviews_for_app(app_name, app_id):
    """
    Fetches reviews for a single app.
    - For 'super.money', it fetches ALL reviews from October 1, 2024.
    - For other apps, it fetches a limited number of recent reviews from May 1, 2024.
    """
    if app_name == 'super.money':
        start_date = datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)
        use_page_limit = False
    else:
        start_date = datetime.datetime(2024, 5, 1, tzinfo=datetime.timezone.utc)
        use_page_limit = True

    all_app_reviews = []
    continuation_token = None
    
    max_pages = 20
    pages_fetched = 0

    while True:
        if use_page_limit and pages_fetched >= max_pages:
            st.info(f"[{app_name}] Reached review limit of {max_pages * 200}. Stopping fetch.")
            break

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
            if use_page_limit:
                pages_fetched += 1
            
            new_reviews = [r for r in result if r['at'].replace(tzinfo=datetime.timezone.utc) >= start_date]
            all_app_reviews.extend(new_reviews)

            if len(new_reviews) < len(result):
                break
            
            if not token:
                break

        except Exception as e:
            st.error(f"An error occurred while fetching reviews for {app_name}: {e}")
            break
    
    if not all_app_reviews:
        return pd.DataFrame()

    df = pd.DataFrame(all_app_reviews)
    df['app'] = app_name
    df = df.rename(columns={'content': 'review', 'at': 'date'})
    return df[['app', 'review', 'date', 'score']].sort_values(by='date', ascending=False).reset_index(drop=True)

def analyze_sentiments_with_vader(df, progress_bar=None, status_text=None):
    """
    Analyzes reviews using VADER with enhanced emoji and misspelling handling.
    """
    if df.empty:
        return df, "No reviews to analyze."

    app_name = df['app'].iloc[0]
    all_sentiments = []
    total_reviews = len(df)
    analyzer = get_vader_analyzer()

    if status_text:
        status_text.text(f"[{app_name}] Analyzing {total_reviews} reviews with VADER...")

    # Enhanced emoji sentiment dictionary
    EMOJI_SENTIMENT = {
        # Positive Emojis
        '😊': 0.8, '😍': 0.9, '😎': 0.7, '😄': 0.85, '😃': 0.85, '😁': 0.8, '😆': 0.8, '😅': 0.7, 
        '😂': 0.8, '🤣': 0.85, '🥰': 0.9, '❤️': 0.9, '💖': 0.85, '✨': 0.6, '🌟': 0.7, '💫': 0.65,
        '🎉': 0.7, '🏆': 0.6, '👏': 0.6, '👍': 0.7, '💪': 0.6, '🔥': 0.65, '🌈': 0.7, '🌸': 0.6,
        
        # Negative Emojis
        '😭': -0.8, '😢': -0.7, '😔': -0.6, '😞': -0.6, '😕': -0.4, '😟': -0.5, '😣': -0.5, 
        '😖': -0.7, '😫': -0.7, '😩': -0.7, '😡': -0.9, '😠': -0.8, '🤬': -0.95, '🤯': -0.6, 
        '😳': -0.3, '😱': -0.7, '😨': -0.7, '😰': -0.6, '😥': -0.5, '😓': -0.5
    }

    def get_emoji_sentiment(text):
        """Calculate sentiment score from emojis in the text."""
        score = 0
        count = 0
        for char in text:
            if char in EMOJI_SENTIMENT:
                score += EMOJI_SENTIMENT[char]
                count += 1
        return score / max(1, count)  # Avoid division by zero

    def get_misspelling_score(text):
        """Calculate misspelling score (lower is better)."""
        # Common misspelling patterns
        misspellings = {
            r'\b(plz|pls|pl0x)\b',
            r'\b(thx|thanx|thnks|thnkx|thnx|thnq|thnku|ty|tysm|tyvm)\b',
            r'\b(u|ur|yr)\b',
            r'\b(r|are)\b',
            r'\b(wont|wont|wont|wont|wont)\b',
            r'\b(dont|dont|dont|dont|dont)\b',
            r'\bcant\b',
            r'\b(im|i m)\b',
            r'\b(ive|i ve)\b',
            r'\b(app|appz|ap|apk)\b',
            r'\b(awesome|awsm|awsum|awsm|awsum)\b',
            r'\b(bad|badd|baddd|badddd|baddddd)\b',
            r'\b(good|gud|gud|gud|gud)\b',
            r'\b(great|gr8|grt|gr8t|gr8test|grate|greatful|grat)\b',
            r'\b(love|luv|lov|lovv|lovve|lovvv|lovvve|lovvved|lovvver|lovvving|lovvved|lovvvving|lovvved)\b',
            r'\b(hate|haet|h8|h8ed|h8ing|h8r|h8rs|h8s|h8ter|h8ters|h8tr|h8trs|h8s)\b',
            r'\b(worst|wrst|w0rst|w0rse)\b'
        }
        
        # Count misspelled words
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0
            
        misspelled = 0
        for word in words:
            for pattern in misspellings:
                if re.fullmatch(pattern, word):
                    misspelled += 1
                    break
        
        return misspelled / len(words)

    for i, row in df.iterrows():
        review_text = str(row['review'])
        review_text_stripped = review_text.strip()

        if not review_text_stripped:
            sentiment = "Inconclusive"
        else:
            # Get base sentiment score
            scores = analyzer.polarity_scores(review_text_stripped)
            compound_score = scores['compound']
            
            # Adjust score based on emojis
            emoji_score = get_emoji_sentiment(review_text_stripped)
            compound_score += emoji_score * 0.3  # Weight emoji score at 30%
            
            # Adjust score based on misspellings
            misspelling_score = get_misspelling_score(review_text_stripped)
            if misspelling_score > 0.2:  # If more than 20% misspellings
                compound_score *= 0.8  # Reduce confidence by 20%
            
            # Determine final sentiment
            if compound_score >= 0.1:  # Adjusted threshold for positive
                sentiment = "Positive"
            elif compound_score <= -0.1:  # Adjusted threshold for negative
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        
        all_sentiments.append(sentiment)
        
        if progress_bar:
            progress_bar.progress((i + 1) / total_reviews)

    df['sentiment'] = all_sentiments

    sentiment_counts = df['sentiment'].value_counts()
    total_sentiments = len(df['sentiment'])

    summary_parts = ["**Sentiment Distribution (VADER):**"]
    categories = ["Positive", "Negative", "Neutral", "Spam", "Inconclusive"]

    for category in categories:
        count = sentiment_counts.get(category, 0)
        percentage = (count / total_sentiments) * 100 if total_sentiments > 0 else 0
        summary_parts.append(f"- **{category}:** {percentage:.1f}% ({count})")

    summary = "  \n".join(summary_parts)
    
    return df, summary
