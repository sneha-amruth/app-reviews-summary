import streamlit as st
import pandas as pd
import datetime as dt
import os
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
from numbers_parser import Document

# Try to import plotly, but make it optional
PLOTLY_AVAILABLE = False

# Simple theme detection
def get_theme():
    """Get the current theme settings"""
    try:
        # Try to get theme from URL parameters using new API
        if hasattr(st, 'query_params') and 'theme' in st.query_params:
            return {'base': st.query_params['theme']}
    except:
        pass
    return {'base': 'light'}

from utils import load_reviews_for_app, analyze_sentiments_with_vader

# Set page config with emoji favicon
st.set_page_config(
    page_title="App Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid rgba(49, 51, 63, 0.2);
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_uploaded_file(uploaded_file):
    """Load data from an uploaded file (CSV or Excel)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def setup_kaggle():
    """Set up Kaggle API with credentials from environment variables."""
    try:
        # Try to get credentials from environment variables
        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')
        
        # If not in environment, try to get from Streamlit secrets
        if not (username and key) and hasattr(st, 'secrets'):
            if 'kaggle' in st.secrets and 'username' in st.secrets['kaggle'] and 'key' in st.secrets['kaggle']:
                username = st.secrets['kaggle']['username']
                key = st.secrets['kaggle']['key']
            elif 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
                username = st.secrets['KAGGLE_USERNAME']
                key = st.secrets['KAGGLE_KEY']
        
        if not (username and key):
            st.error("Kaggle API credentials not found.")
            st.info("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables or add them to Streamlit secrets.")
            return False
            
        # Set environment variables
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        return True
        
    except Exception as e:
        st.error(f"Error setting up Kaggle: {str(e)}")
        return False

def fetch_and_load_kaggle_dataset(dataset_slug, download_path='/tmp/kaggle_data'):
    """Downloads a dataset from Kaggle, unzips it, and loads data from the first CSV or Excel file found."""
    if not KAGGLE_AVAILABLE:
        st.error("Kaggle API is not available in this environment.")
        return None
        
    # Set up Kaggle
    if not setup_kaggle():
        return None
        
    # Clean up download directory
    import shutil
    if os.path.exists(download_path):
        shutil.rmtree(download_path)
    os.makedirs(download_path)

    try:
        api = KaggleApi()
        api.authenticate()
        
        st.info(f"Downloading dataset '{dataset_slug}' from Kaggle...")
        api.dataset_download_files(dataset_slug, path=download_path, unzip=True, quiet=False)
        st.info("Download complete. Loading data...")

        # Debug: List all files in the download directory
        all_files = []
        for root, dirs, files in os.walk(download_path):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        st.info(f"Found {len(all_files)} files in the dataset.")
        
        # Try to find and load data files
        for file_path in all_files:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == '.csv':
                    df = pd.read_csv(file_path)
                elif file_ext in ['.xls', '.xlsx']:
                    df = pd.read_excel(file_path)
                elif file_ext == '.json':
                    df = pd.read_json(file_path)
                elif file_ext == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_ext == '.numbers':
                    # Handle Apple Numbers file
                    from numbers_parser import Document
                    doc = Document(file_path)
                    sheets = doc.sheets
                    tables = sheets[0].tables
                    rows = tables[0].rows()
                    
                    # Convert to DataFrame
                    data = []
                    for row in rows[1:]:  # Skip header
                        data.append([cell.value for cell in row])
                    
                    if not data:
                        st.warning(f"No data found in {os.path.basename(file_path)}")
                        continue
                        
                    df = pd.DataFrame(
                        data,
                        columns=[str(cell.value) for cell in rows[0]]  # Use first row as header
                    )
                else:
                    continue
                
                if not df.empty:
                    st.success(f"Successfully loaded data from '{os.path.basename(file_path)}' with shape {df.shape}")
                    return df
                
            except Exception as e:
                st.warning(f"Could not read {file_path}: {str(e)}")
                continue
        
        # If we get here, no files were successfully loaded
        st.error("No compatible data files found in the dataset. Here are the files we found:")
        for i, file_path in enumerate(all_files, 1):
            st.write(f"{i}. {file_path}")
            
        return None
        
    except Exception as e:
        st.error(f"Error accessing Kaggle: {str(e)}")
        st.info("Please check:")
        st.info("1. The dataset name is correct")
        st.info("2. You have access to the dataset (some are private)")
        st.info("3. The dataset contains supported file types (CSV, Excel, JSON, Parquet)")
        return None

# App title
st.title("📱 App Review Analyzer")

# Kaggle Dataset Input
st.markdown("### Load Data from Kaggle")

# Example dataset (you can change this default)
default_dataset = "lava18/google-play-store-apps"
dataset_slug = st.text_input(
    "Enter Kaggle dataset (e.g., 'username/dataset-name')",
    value=default_dataset,
    help="The dataset should be in CSV or Excel format with app review data"
)

if st.button("Load from Kaggle") and dataset_slug:
    with st.spinner("Fetching data from Kaggle..."):
        reviews_df = fetch_and_load_kaggle_dataset(dataset_slug)
        if reviews_df is not None:
            st.session_state['current_reviews'] = reviews_df
            st.session_state['current_app'] = f"Kaggle: {dataset_slug}"

# Add a divider
st.markdown("---")

# Sidebar with app selection
st.sidebar.title("📋 App Selection")
selected_app = st.sidebar.selectbox(
    "Choose an app to analyze:",
    ['Navi', 'Kiwi', 'super.money', 'Pop'],
    index=0
)

# App configurations
APPS = {
    'Navi': {
        'id': 'com.naviapp',
        'source': 'kaggle',
        'dataset': 'snehaamruth/navi-play-store-reviews-may-2024-to-jun-2025'
    },
    'Kiwi': {
        'id': 'in.gokiwi.kiwitpap',
        'source': 'kaggle',
        'dataset': 'snehaamruth/kiwi-play-store-reviews-may-2024-to-jun-2025'
    },
    'Pop': {
        'id': 'com.popclub.android',
        'source': 'kaggle',
        'dataset': 'snehaamruth/pop-play-store-reviews-may-2024-to-jun-2025'
    },
    'super.money': {
        'id': 'money.super.payments',
        'source': 'playstore'
    }
}

# Main content area
app_config = APPS[selected_app]
app_id = app_config['id']

# Get theme info
is_dark = get_theme().get('base', 'light') == 'dark'

st.markdown(f"## {selected_app}")
st.markdown("---")

# Fetch button
if st.button("🔄 Fetch Reviews", key=f"fetch_{app_id}", type="primary"):
        with st.spinner("🔄 Fetching reviews... This may take a moment..."):
            try:
                if app_config['source'] == 'kaggle':
                    reviews_df = fetch_and_load_kaggle_dataset(app_config['dataset'])
                else:
                    reviews_df = load_reviews_for_app(selected_app, app_id)
                
                if not reviews_df.empty:
                    # Convert date column if it exists
                    if 'at' in reviews_df.columns:
                        reviews_df['date'] = pd.to_datetime(reviews_df['at']).dt.date
                    elif 'date' in reviews_df.columns:
                        reviews_df['date'] = pd.to_datetime(reviews_df['date']).dt.date
                    
                    st.session_state['current_reviews'] = reviews_df
                    st.session_state['current_app'] = selected_app
                    st.rerun()
                else:
                    st.error("No reviews found. Please try again later.")
            except Exception as e:
                st.error(f"Error fetching reviews: {str(e)}")

# Show analysis section if reviews are loaded
if 'current_reviews' in st.session_state and st.session_state['current_app'] == selected_app:
    reviews_df = st.session_state['current_reviews']
    
    # Date filtering removed for simplicity
    
    # Review stats
    st.caption(f"📊 {len(reviews_df):,} reviews")
    
    # Analysis button
    if st.button("✨ Analyze Sentiment", type="primary"):
        st.session_state['analyze_clicked'] = True
        
        if 'analyze_clicked' in st.session_state and st.session_state['analyze_clicked']:
            with st.spinner("🔍 Analyzing sentiment (this may take a moment)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    analyzed_df, summary = analyze_sentiments_with_vader(reviews_df, progress_bar, status_text)
                    st.session_state['analysis_results'] = (analyzed_df, summary)
                    st.session_state['analyze_clicked'] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                finally:
                    progress_bar.empty()
    
    # Show analysis results if available
    if 'analysis_results' in st.session_state:
        results_df, summary = st.session_state['analysis_results']
        
        # Sentiment results
        st.markdown("### Sentiment Analysis")
        
        # Show sentiment distribution
        sentiment_counts = results_df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Show counts in a compact way
        cols = st.columns(4)
        for i, (sentiment, count) in enumerate(sentiment_counts.items()):
            with cols[i % 4]:
                st.metric(sentiment, f"{count:,}")
        
        # Show summary
        with st.expander("View Summary"):
            st.markdown(summary)
        
        # Download button
        st.download_button(
            label="💾 Download Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_app.lower().replace(' ', '_')}_analysis_{dt.date.today().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
        # Detailed results in expander
        with st.expander("📋 View Detailed Results", expanded=False):
            st.dataframe(
                results_df,
                column_config={
                    'review': st.column_config.TextColumn("Review", width="large"),
                    'sentiment': st.column_config.TextColumn("Sentiment"),
                    'score': st.column_config.NumberColumn("Rating", format="%d ⭐")
                },
                hide_index=True,
                use_container_width=True
            )
else:
    # Empty state
    st.info("Click 'Fetch Reviews' to begin analysis")
