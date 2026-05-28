# =========================
# ✅ 1. CRITICAL THREAD LOCKS (Must be first)
# =========================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from transformers import pipeline
import torch
import gc

# ✅ 2. PYTORCH RAM & THREAD LOCK
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False) # Massive RAM saver: prevents PyTorch from storing memory for model training

st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# =========================
# Local Model 
# =========================
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    # Standard loading (without accelerate/low_cpu_mem_usage to avoid semaphore crashes)
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

with st.spinner("Loading AI Model into memory..."):
    sentiment_model = load_sentiment_model()

# =========================
# Core Data Fetching
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker_symbol):
    url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        return feedparser.parse(url).entries[:15]
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_price(ticker_symbol):
    try:
        # ✅ 3. YFINANCE FIX: threads=False stops the semaphore leaks!
        return yf.download(ticker_symbol, period="7d", interval="1h", threads=False)
    except Exception:
        return pd.DataFrame()

with st.spinner("Fetching market data..."):
    news = get_news(ticker)
    price = get_price(ticker)

if not news or price.empty:
    st.warning("Data unavailable for this ticker.")
    st.stop()

# =========================
# Local Sentiment Analysis
# =========================
data = []
with st.spinner("Analyzing Sentiment..."):
    week_ago = datetime.now().date() - timedelta(days=7)
    
    for entry in news:
        if not hasattr(entry, "published_parsed"): continue
        date = datetime(*entry.published_parsed[:6]).date()
        if date < week_ago: continue
            
        result = sentiment_model(entry.title)[0]
        label = result['label']
        score = 1 if label == "positive" else -1 if label == "negative" else 0
        
        data.append({
            "date": date, 
            "label": label, 
            "score": score, 
            "title": entry.title,
            "link": getattr(entry, 'link', '#') 
        })
        
    gc.collect()

df = pd.DataFrame(data)
if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

# =========================
# Sentiment Counting Indicator
# =========================
st.markdown("### 📊 Sentiment Overview")
pos_count = len(df[df['label'] == 'positive'])
neg_count = len(df[df['label'] == 'negative'])
neu_count = len(df[df['label'] == 'neutral'])

col1, col2, col3 = st.columns(3)
col1.metric("🟢 Positive News", pos_count)
col2.metric("🔴 Negative News", neg_count)
col3.metric("⚪ Neutral News", neu_count)

# =========================
# Charts & Visuals
# =========================
df_daily = df.groupby("date")["score"].mean().rolling(2, min_periods=1).mean()

close_col = price["Close"]
if isinstance(close_col, pd.DataFrame):
    close_col = close_col.iloc[:, 0]
price_daily = close_col.groupby(price.index.date).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=price_daily.index, y=price_daily.values, 
    mode='lines+markers', name="Stock Price", line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=df_daily.index, y=df_daily.values, 
    mode='lines+markers', name="Sentiment Score", yaxis="y2", line=dict(color='orange')
))

fig.update_layout(
    title=f"{ticker} Price vs Sentiment Trend",
    yaxis2=dict(overlaying="y", side="right", showgrid=False),
    height=400, margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Rule-Based Market Summary
# =========================
st.markdown("### 🧠 Market Summary")
trend = "positive" if df_daily.iloc[-1] > 0 else "negative" if df_daily.iloc[-1] < 0 else "neutral"

pos_headlines = df[df['label'] == 'positive']['title'].tolist()
neg_headlines = df[df['label'] == 'negative']['title'].tolist()

if trend == "positive" and pos_headlines:
    summary = f"Market sentiment is structurally positive. Key catalyst: '{pos_headlines[0]}'"
elif trend == "negative" and neg_headlines:
    summary = f"Market sentiment leans negative, driven primarily by risks such as: '{neg_headlines[0]}'"
else:
    summary = f"Market sentiment is highly mixed and currently tracking {trend}."

st.success(summary)

# =========================
# News Feed
# =========================
st.markdown("### 📰 Recent Headlines")
for _, row in df.head(10).iterrows():
    emoji = "🟢" if row["label"] == "positive" else "🔴" if row["label"] == "negative" else "⚪"
    st.markdown(f"{emoji} [{row['title']}]({row['link']})")
