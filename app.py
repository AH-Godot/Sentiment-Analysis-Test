# =========================
# 1. CRITICAL THREAD & RESOURCE LOCKS
# =========================
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Lock PyTorch memory before it can spawn background threads
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False) # Prevents storing unnecessary training tensors

import gc

# =========================
# 2. STANDARD IMPORTS
# =========================
import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# =========================
# 3. INTERACTIVE TRIGGER (Prevents Boot-Up Crashes)
# =========================
if st.button("🚀 Run Local AI Analysis"):
    
    # --- FETCH DATA ---
    with st.spinner("Fetching market data..."):
        url = f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en"
        try:
            news = feedparser.parse(url).entries[:15] # Limit to 15 to save memory
        except Exception:
            news = []
        
        try:
            # threads=False prevents semaphore leaks
            price = yf.download(ticker, period="7d", interval="1h", threads=False) 
        except Exception:
            price = pd.DataFrame()
            
    if not news or price.empty:
        st.warning("Data unavailable for this ticker.")
        st.stop()

    # --- LOAD, PREDICT, AND DELETE ---
    data = []
    week_ago = datetime.now().date() - timedelta(days=7)
    
    with st.spinner("Loading Distilled AI Model (Local)..."):
        # ✅ Using a smaller, distilled financial model to save RAM
        sentiment_model = pipeline(
            "sentiment-analysis", 
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )
        
        with st.spinner("Analyzing Sentiments..."):
            for entry in news:
                if not hasattr(entry, "published_parsed"): continue
                dt = datetime(*entry.published_parsed[:6]).date()
                if dt < week_ago: continue
                    
                result = sentiment_model(entry.title)[0]
                label = result['label'].lower()
                
                score = 1 if label == "positive" else -1 if label == "negative" else 0
                
                data.append({
                    "date": dt, 
                    "label": label, 
                    "score": score, 
                    "title": entry.title,
                    "link": getattr(entry, 'link', '#') 
                })
        
        # 🔴 CRITICAL MEMORY SAVER: Delete model from RAM instantly
        del sentiment_model
        gc.collect()

    # --- BUILD UI ---
    df = pd.DataFrame(data)
    if df.empty:
        st.warning("No recent sentiment data available.")
        st.stop()

    # Metrics
    st.markdown("### 📊 Sentiment Overview")
    cols = st.columns(3)
    cols[0].metric("🟢 Positive News", len(df[df['label'] == 'positive']))
    cols[1].metric("🔴 Negative News", len(df[df['label'] == 'negative']))
    cols[2].metric("⚪ Neutral News", len(df[df['label'] == 'neutral']))

    # Charts
    df_daily = df.groupby("date")["score"].mean().rolling(2, min_periods=1).mean()

    # Flatten yfinance MultiIndex safely
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
    
    # Market Summary
    st.markdown("### 🧠 AI Market Summary")
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

    # News Feed
    st.markdown("### 📰 Recent Headlines")
    for _, row in df.head(10).iterrows():
        emoji = "🟢" if row["label"] == "positive" else "🔴" if row["label"] == "negative" else "⚪"
        st.markdown(f"{emoji} [{row['title']}]({row['link']})")
