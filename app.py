import streamlit as st
import pandas as pd
from utils import load_reviews_for_app, analyze_reviews_with_vader
import datetime

st.set_page_config(page_title="App Review Analysis", layout="wide")

st.title("App Review Analysis Workflow")
st.write("Follow the two-step process to fetch, download, and analyze reviews.")

APPS = {
    'Navi': 'com.naviapp',
    'Kiwi': 'in.gokiwi.kiwitpap',
    'super.money': 'money.super.payments',
    'Pop': 'com.popclub.android'
}

for app_name, app_id in APPS.items():
    with st.expander(f"Process Reviews for {app_name}", expanded=True):
        st.subheader(f"Step 1: Fetch Reviews for {app_name}")
        if st.button(f"Fetch Latest Reviews", key=f"fetch_{app_id}"):
            with st.spinner(f"Fetching reviews for {app_name}..."):
                reviews_df = load_reviews_for_app(app_name, app_id)
                if reviews_df.empty:
                    st.warning(f"No new reviews found for {app_name} since October 2024.")
                    if f'fetched_reviews_{app_id}' in st.session_state:
                        del st.session_state[f'fetched_reviews_{app_id}']
                else:
                    st.success(f"Successfully fetched {len(reviews_df)} reviews.")
                    st.session_state[f'fetched_reviews_{app_id}'] = reviews_df

        if f'fetched_reviews_{app_id}' in st.session_state:
            reviews_df = st.session_state[f'fetched_reviews_{app_id}']
            st.subheader(f"Step 2: Download or Analyze")
            
            st.download_button(
                label=f"Download {len(reviews_df)} Reviews (CSV)",
                data=reviews_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{app_name}_reviews_{datetime.date.today()}.csv",
                mime='text/csv',
                key=f"download_{app_id}"
            )

            if st.button(f"Analyze {len(reviews_df)} Reviews", key=f"analyze_{app_id}"):
                st.info(f"Analyzing reviews for {app_name}. This may take a few minutes...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_df, summary = analyze_reviews_with_vader(
                    reviews_df,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                st.session_state[f'analysis_results_{app_id}'] = (results_df, summary)
                status_text.success(f"Analysis complete for {app_name}!")
                progress_bar.empty()

        if f'analysis_results_{app_id}' in st.session_state:
            results_df, summary = st.session_state[f'analysis_results_{app_id}']
            st.subheader(f"Analysis Results for {app_name}")
            st.write(summary)
            st.dataframe(results_df)
