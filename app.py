# =========================
# ✅ Anti-Crash Settings (Must be first)
# =========================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_VISUAL_BACKENDS"] = "1"

import torch
torch.set_num_threads(1)

# =========================
# Imports
# =========================
import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
from transformers import pipeline
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import requests

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# =========================
# ✅ Load Model 1 (Local Pipeline)
# =========================
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    # Pipeline 1: Local Fine-tuned Sentiment Model
    # Note: Replace "ProsusAI/finbert" with your own HF model path once uploaded
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

with st.spinner("Loading Sentiment Model..."):
    sentiment_model = load_sentiment_model()

# =========================
# Data Fetching
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker_symbol):
    try:
        url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
        return feedparser.parse(url).entries
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_price(ticker_symbol):
    try:
        return yf.download(ticker_symbol, period="7d", interval="1h")
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()

with st.spinner(f"Fetching data for {ticker}..."):
    news = get_news(ticker)
    price = get_price(ticker)

if not news or price.empty:
    st.warning("Data unavailable for this ticker. Please try another symbol.")
    st.stop()

# =========================
# Sentiment Analysis
# =========================
week_ago = datetime.now().date() - timedelta(days=7)

data = []
with st.spinner("Analyzing Sentiment..."):
    for entry in news[:50]:
        if not hasattr(entry, "published_parsed"):
            continue

        date = datetime(*entry.published_parsed[:6]).date()
        if date < week_ago:
            continue

        title = entry.title
        result = sentiment_model(title)[0]
        score = 1 if result["label"] == "positive" else -1 if result["label"] == "negative" else 0

        data.append({
            "date": date,
            "label": result["label"],
            "score": score,
            "title": title
        })

df = pd.DataFrame(data)

if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

# =========================
# Time Series Processing
# =========================
df_daily = df.groupby("date")["score"].mean().rolling(2).mean()
price["date"] = price.index.date
price_daily = price.groupby("date")["Close"].mean()

# =========================
# Trading Signal
# =========================
def generate_signal(sentiment_daily, stock_daily):
    merged = pd.merge(stock_daily, sentiment_daily, left_index=True, right_index=True).dropna()

    if len(merged) < 3:
        return "HOLD", 0

    corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
    trend = sentiment_daily.tail(3).mean()

    if trend > 0.3 and corr > 0.2:
        return "BUY", corr
    elif trend < -0.3 and corr > 0.2:
        return "SELL", corr
    else:
        return "HOLD", corr

signal, corr = generate_signal(df_daily, price_daily)

st.markdown("## 🚦 Trading Signal")
if signal == "BUY":
    st.success(f"🟢 **BUY** | Correlation: {corr:.2f}")
elif signal == "SELL":
    st.error(f"🔴 **SELL** | Correlation: {corr:.2f}")
else:
    st.info(f"⚪ **HOLD** | Correlation: {corr:.2f}")

# =========================
# Charts
# =========================
col1, col2 = st.columns([1,2])

with col1:
    counts = df["label"].value_counts()
    fig_sent = go.Figure()
    fig_sent.add_bar(
        x=counts.index,
        y=counts.values,
        marker_color=["green" if x=="positive" else "red" if x=="negative" else "gray" for x in counts.index]
    )
    fig_sent.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title="Sentiment Breakdown")
    st.plotly_chart(fig_sent, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily, name="Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily, name="Sentiment", yaxis="y2", line=dict(color='orange')))

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Price vs. Sentiment Trend",
        yaxis2=dict(overlaying="y", side="right", showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TF-IDF Keywords & Sentiment
# =========================
def extract_keywords(texts):
    vec = TfidfVectorizer(stop_words="english", max_features=50)
    X = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    words = vec.get_feature_names_out()
    return sorted(zip(words, scores), key=lambda x: x[1], reverse=True)[:10]

keywords = extract_keywords(df["title"].tolist())

keyword_sentiment = {}
for _, row in df.iterrows():
    for word, _ in keywords:
        if word in row["title"].lower():
            keyword_sentiment.setdefault(word, []).append(row["score"])

keyword_sentiment = {k: sum(v)/len(v) for k, v in keyword_sentiment.items()}

# =========================
# Word Cloud & Market Themes
# =========================
st.markdown("### 🔑 Market Themes")
colA, colB = st.columns(2)

with colA:
    def get_color(word):
        val = keyword_sentiment.get(word, 0)
        return "green" if val > 0 else "red" if val < 0 else "gray"

    wc = WordCloud(width=500, height=250, background_color="white")
    wc.generate_from_frequencies(dict(keywords))
    wc.recolor(color_func=lambda word, *args, **kwargs: get_color(word))

    fig_wc, ax = plt.subplots(figsize=(5,3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

with colB:
    ks_df = pd.DataFrame(keyword_sentiment.items(), columns=["Keyword", "Sentiment Score"])
    st.bar_chart(ks_df.set_index("Keyword"))

# =========================
# ✅ Pipeline 2: API Generation
# =========================
st.markdown("### 🧠 Market Summary")

def call_hf_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
    
    # TIP: For better reliability, get a free HF Token and add it to Streamlit secrets.
    # headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    headers = {} 
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 40, "return_full_text": False, "do_sample": False}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return None
    except Exception:
        return None

def generate_summary(titles, keywords_list, kw_sentiment, sig):
    top_kw = [k for k, _ in keywords_list[:3]]
    pos_kw = [k for k, v in kw_sentiment.items() if v > 0]
    neg_kw = [k for k, v in kw_sentiment.items() if v < 0]
    
    tone = "positive" if sig == "BUY" else "negative" if sig == "SELL" else "neutral"
    input_text = " ".join(titles[:5])[:300]
    keyword_hint = ", ".join(top_kw)

    prompt = f"""
You are a financial analyst.
Write ONE professional sentence summarizing market sentiment.
Tone: {tone}
Focus on: {keyword_hint}
News: {input_text}
Answer:
"""
    
    # Attempt Pipeline 2 (API Call)
    sentence = None
    with st.spinner("Generating AI Summary..."):
        generated_text = call_hf_api(prompt)
        
        if generated_text:
            text = generated_text.replace(prompt, "").strip()
            match = re.search(r"(.+?\.)", text)
            sentence = match.group(1) if match else text[:200]
            
    # Fallback if API is asleep or rate-limited
    if not sentence:
        pos = ", ".join(pos_kw[:2]) or "limited catalysts"
        neg = ", ".join(neg_kw[:2]) or "minor risks"

        if sig == "BUY":
            sentence = f"Market sentiment is structurally positive, supported by {pos}, though residual risks remain around {neg}."
        elif sig == "SELL":
            sentence = f"Market sentiment leans negative, driven primarily by {neg}, with limited support currently visible from {pos}."
        else:
            sentence = f"Market sentiment is highly mixed; support from {pos} is currently being offset by emerging risks in {neg}."

    extra_info = f"""
✅ **Key Opportunities:** {', '.join(pos_kw[:3]) or 'None identified'}  
⚠️ **Primary Risks:** {', '.join(neg_kw[:3]) or 'None identified'}
"""
    return sentence, extra_info

summary, extra = generate_summary(df["title"].tolist(), keywords, keyword_sentiment, signal)

st.success(summary)
st.markdown(extra)

# =========================
# News Feed
# =========================
st.markdown("### 📰 Recent Headlines")
for entry in news[:10]:
    st.markdown(f"• [{entry.title}]({entry.link})")
