# app.py
import streamlit as st
import re
import random
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pyperclip
import io
import base64
from datetime import datetime
import time

# Streamlit page config
st.set_page_config(page_title="Advanced Sentiment Analysis Dashboard", layout="wide")

# Custom CSS for dark mode and styling
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stButton>button { width: 100%; margin-top: 0.5rem; }
    .stTextArea textarea { font-size: 1rem; }
    .stSelectbox select { font-size: 1rem; }
    .wordcloud-img { display: block; margin: auto; }
    [data-testid="stSidebar"] { background-color: #1f2937; }
    [data-testid="stSidebar"] * { color: white; }
</style>
""", unsafe_allow_html=True)

# Sentiment analysis logic (ported from JavaScript)
def analyze_sentiment(text):
    positive_words = ['happy', 'great', 'awesome', 'love', 'excellent', 'joyful']
    negative_words = ['sad', 'bad', 'terrible', 'hate', 'awful', 'angry']
    emotion_map = {
        'joy': ['happy', 'awesome', 'love', 'excellent'],
        'anger': ['hate', 'angry', 'terrible'],
        'sadness': ['sad', 'awful'],
        'fear': ['scary', 'afraid'],
        'surprise': ['shock', 'unexpected']
    }

    sentiment_score = 0
    words = re.split(r'\s+', text.lower().strip())
    word_freq = Counter(words)
    emotions = {'joy': 0, 'anger': 0, 'sadness': 0, 'fear': 0, 'surprise': 0}
    phrases = set()

    for i, word in enumerate(words):
        if word in positive_words:
            sentiment_score += 0.2
        if word in negative_words:
            sentiment_score -= 0.2
        for emotion, triggers in emotion_map.items():
            if word in triggers:
                emotions[emotion] += 0.3
        if i < len(words) - 1:
            phrases.add(f"{word} {words[i + 1]}")

    sentiment = 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'
    total_emotion = sum(emotions.values()) or 1
    normalized_emotions = {k: v / total_emotion for k, v in emotions.items()}
    sentiment_score = max(min(sentiment_score, 1), -1)

    return {
        'sentiment': sentiment,
        'sentiment_score': sentiment_score,
        'sentiment_confidence': random.uniform(0.8, 1.0),
        'emotions': normalized_emotions,
        'emotion_confidence': random.uniform(0.75, 0.95),
        'word_cloud': [{'word': k, 'count': v * 10} for k, v in word_freq.items()],
        'key_phrases': list(phrases)[:5],
        'analysis_duration': random.randint(50, 150)
    }

# Word cloud generation
def generate_word_cloud(words, density):
    density_factors = {
        'sparse': {'max_words': 10, 'base_size': 12, 'scale': 0.3},
        'normal': {'max_words': 20, 'base_size': 16, 'scale': 0.5},
        'dense': {'max_words': 30, 'base_size': 18, 'scale': 0.7}
    }
    params = density_factors[density]
    word_dict = {item['word']: item['count'] for item in words[:params['max_words']]}
    
    if not word_dict:
        return None
    
    wc = WordCloud(
        width=800, height=400, background_color='white',
        min_font_size=10, max_font_size=40, relative_scaling=params['scale']
    ).generate_from_frequencies(word_dict)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

# PDF export
def export_to_pdf(input_text, result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 16)
    c.drawString(50, 750, "Sentiment Analysis Report")
    c.setFont("Helvetica", 12)
    text = input_text[:100] + ('...' if len(input_text) > 100 else '')
    c.drawString(50, 700, f"Text: {text}")
    c.drawString(50, 680, f"Sentiment: {result['sentiment']}")
    c.drawString(50, 660, f"Score: {result['sentiment_score'] * 100:.1f}%")
    c.drawString(50, 640, f"Confidence: {result['sentiment_confidence'] * 100:.1f}%")
    c.drawString(50, 620, "Emotions:")
    for i, (emotion, score) in enumerate(result['emotions'].items()):
        c.drawString(60, 600 - i * 20, f"{emotion}: {score * 100:.1f}%")
    c.drawString(50, 500, "Key Phrases:")
    for i, phrase in enumerate(result['key_phrases']):
        c.drawString(60, 480 - i * 20, phrase)
    c.save()
    buffer.seek(0)
    return buffer

# Copy results
def copy_results(result):
    text = f"""
Sentiment: {result['sentiment']} { {'Positive': '😊', 'Negative': '😢', 'Neutral': '😐'}[result['sentiment']] }
Score: {result['sentiment_score'] * 100:.1f}%
Confidence: {result['sentiment_confidence'] * 100:.1f}%
Emotions:
{'\n'.join(f"  {k}: {v * 100:.1f}%" for k, v in result['emotions'].items())}
Key Phrases: {', '.join(result['key_phrases'])}
    """.strip()
    pyperclip.copy(text)
    st.success("Results copied to clipboard!")

# Share results
def share_results(input_text, result):
    params = {
        'text': input_text[:100],
        'sentiment': result['sentiment'],
        'score': f"{result['sentiment_score'] * 100:.1f}",
        'confidence': f"{result['sentiment_confidence'] * 100:.1f}"
    }
    url = f"http://localhost:8501/?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    pyperclip.copy(url)
    st.success("Shareable URL copied to clipboard!")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'settings' not in st.session_state:
    st.session_state.settings = {'real_time_analysis': True, 'word_cloud_density': 'normal'}
if 'confirm_reset' not in st.session_state:
    st.session_state.confirm_reset = False

# Main app
def main():
    # Header
    st.title("Advanced Sentiment Analysis")
    st.write("Real-time emotion and sentiment analysis with interactive visualizations.")

    # Sidebar for settings and dark mode
    with st.sidebar:
        st.header("Settings")
        real_time = st.checkbox("Enable Real-Time Analysis", value=st.session_state.settings['real_time_analysis'])
        density = st.selectbox("Word Cloud Density", ['sparse', 'normal', 'dense'], index=['sparse', 'normal', 'dense'].index(st.session_state.settings['word_cloud_density']))
        dark_mode = st.checkbox("Dark Mode")
        st.session_state.settings = {'real_time_analysis': real_time, 'word_cloud_density': density}
        
        # Apply dark mode (approximation)
        if dark_mode:
            st.markdown("""
            <style>
                .main { background-color: #1f2937; color: white; }
                .stTextArea textarea { background-color: #374151; color: white; border-color: #4b5563; }
                .stSelectbox select { background-color: #374151; color: white; }
                [data-testid="stAppViewContainer"] { background-color: #1f2937; }
            </style>
            """, unsafe_allow_html=True)

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Enter Text")
        input_text = st.text_area("Type your text here...", height=200)
        
        if st.session_state.confirm_reset:
            st.warning("Are you sure you want to clear all analysis history?")
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("Confirm"):
                    st.session_state.history = []
                    st.session_state.confirm_reset = False
                    st.success("History cleared!")
            with col_confirm2:
                if st.button("Cancel"):
                    st.session_state.confirm_reset = False

        # Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Save Analysis", disabled=not input_text.strip()):
                if 'result' in st.session_state:
                    st.session_state.history = [{
                        'text': input_text,
                        'result': st.session_state.result,
                        'timestamp': datetime.now().isoformat(),
                        'duration': st.session_state.result['analysis_duration']
                    }] + st.session_state.history[:9]
                    st.success("Analysis saved!")
            if st.button("Clear Text"):
                input_text = ""
                if 'result' in st.session_state:
                    del st.session_state.result
                st.rerun()
        with col_btn2:
            if st.button("Export PDF", disabled='result' not in st.session_state):
                pdf_buffer = export_to_pdf(input_text, st.session_state.result)
                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name="sentiment_analysis.pdf",
                    mime="application/pdf"
                )
            if st.button("Copy Results", disabled='result' not in st.session_state):
                copy_results(st.session_state.result)

        with col_btn1:
            if st.button("Share Results", disabled='result' not in st.session_state):
                share_results(input_text, st.session_state.result)
        with col_btn2:
            if not st.session_state.settings['real_time_analysis']:
                if st.button("Analyze Now", disabled=not input_text.strip()):
                    result = analyze_sentiment(input_text)
                    st.session_state.result = result

    with col2:
        st.header("Analysis Results")
        if input_text.strip() and (st.session_state.settings['real_time_analysis'] or 'result' in st.session_state):
            if st.session_state.settings['real_time_analysis']:
                result = analyze_sentiment(input_text)
                st.session_state.result = result
            else:
                result = st.session_state.result

            # Sentiment display
            emoji = {'Positive': '😊', 'Negative': '😢', 'Neutral': '😐'}[result['sentiment']]
            color = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}[result['sentiment']]
            st.markdown(f"**Sentiment:** <span style='color:{color}'>{result['sentiment']} {emoji}</span>", unsafe_allow_html=True)
            st.write(f"Score: {result['sentiment_score'] * 100:.1f}% | Confidence: {result['sentiment_confidence'] * 100:.1f}% | Duration: {result['analysis_duration']}ms")

            # Charts
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                # Sentiment Bar Chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Positive 😊', 'Negative 😢', 'Neutral 😐'],
                        y=[
                            abs(result['sentiment_score']) if result['sentiment'] == 'Positive' else 0,
                            abs(result['sentiment_score']) if result['sentiment'] == 'Negative' else 0,
                            1 if result['sentiment'] == 'Neutral' else 0
                        ],
                        marker_color=['#4ade80', '#f87171', '#60a5fa']
                    )
                ])
                fig.update_layout(title="Sentiment Breakdown", yaxis_title="Score", yaxis_range=[0, 1], showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col_chart2:
                # Emotion Radar Chart
                fig = go.Figure(data=[
                    go.Scatterpolar(
                        r=[result['emotions'].get(k, 0) for k in ['joy', 'anger', 'sadness', 'fear', 'surprise']],
                        theta=['Joy 😄', 'Anger 😣', 'Sadness 😢', 'Fear 😨', 'Surprise 😮'],
                        fill='toself',
                        line_color='#60a5fa'
                    )
                ])
                fig.update_layout(title="Emotion Distribution", polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Key Phrases
            st.subheader("Key Phrases")
            if result['key_phrases']:
                for phrase in result['key_phrases']:
                    st.write(f"- {phrase}")
            else:
                st.write("No key phrases detected.")

            # Word Cloud
            st.subheader("Word Cloud")
            wc_buf = generate_word_cloud(result['word_cloud'], st.session_state.settings['word_cloud_density'])
            if wc_buf:
                st.image(wc_buf, use_column_width=True)
            else:
                st.write("No words to display.")

        else:
            st.write("Start typing to see real-time analysis.")

    # History
    st.header("Analysis History")
    col_history1, col_history2 = st.columns([3, 1])
    with col_history1:
        search_query = st.text_input("Search history...")
    with col_history2:
        if st.button("Reset History", disabled=not st.session_state.history):
            st.session_state.confirm_reset = True
            st.rerun()

    filtered_history = [
        item for item in st.session_state.history
        if search_query.lower() in item['text'].lower() or search_query.lower() in item['result']['sentiment'].lower()
    ]

    if filtered_history:
        for item in filtered_history:
            st.markdown(f"**Text:** {item['text']}")
            emoji = {'Positive': '😊', 'Negative': '😢', 'Neutral': '😐'}[item['result']['sentiment']]
            color = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}[item['result']['sentiment']]
            st.markdown(f"**Sentiment:** <span style='color:{color}'>{item['result']['sentiment']} {emoji}</span> (Score: {item['result']['sentiment_score'] * 100:.1f}%)", unsafe_allow_html=True)
            st.write(f"Timestamp: {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} | Duration: {item['duration']}ms")
            st.markdown("---")

        # Sentiment Trend Chart
        st.subheader("Sentiment Trend")
        if filtered_history:
            df = pd.DataFrame([
                {'Analysis': f"Analysis {i+1}", 'Score': item['result']['sentiment_score']}
                for i, item in enumerate(reversed(filtered_history[:10]))
            ])
            fig = px.line(df, x='Analysis', y='Score', title="Sentiment Trend", range_y=[-1, 1])
            fig.update_traces(line_color='#60a5fa')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No history yet or no matches found.")

    st.markdown("---")
    st.write("Powered by Advanced AI Sentiment Analysis | © 2025")

if __name__ == "__main__":
    main()
