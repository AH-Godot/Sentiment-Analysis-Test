import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# =========================
# 1. SETUP & RESILIENT NETWORK SESSION
# =========================
st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# Add your Hugging Face Token in Streamlit Cloud Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ✅ Resilient Session for Hugging Face (Prevents DNS / Network Drop Errors)
def get_resilient_session():
    session = requests.Session()
    retry = Retry(total=5, connect=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

http_session = get_resilient_session()

# =========================
# 2. FETCH DATA (With Anti-Block Bypass)
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker_symbol):
    # Fetch News
    url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        news = feedparser.parse(url).entries[:15]
    except Exception:
        news = []
    
    # Fetch Price (Disguised as a real browser)
    try:
        yf_session = requests.Session()
        yf_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        })
        stock = yf.Ticker(ticker_symbol, session=yf_session)
        price = stock.history(period="7d", interval="1h")
    except Exception:
        price = pd.DataFrame()
        
    return news, price

with st.spinner("Fetching market data..."):
    news, price = fetch_market_data(ticker)
    
if not news or price.empty:
    st.warning("Data unavailable for this ticker.")
    st.stop()

# =========================
# 3. API SENTIMENT ANALYSIS
# =========================
week_ago = datetime.now().date() - timedelta(days=7)
valid_news = [n for n in news if hasattr(n, "published_parsed") and datetime(*n.published_parsed[:6]).date() >= week_ago]
titles = [n.title for n in valid_news]
labels = ["neutral"] * len(titles) # Fallback

with st.spinner("Analyzing sentiments via Hugging Face API..."):
    if titles:
        # ✅ Using the distilled model via API for speed
        API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        try:
            response = http_session.post(API_URL, headers=HEADERS, json={"inputs": titles}, timeout=30)
            if response.status_code == 200:
                results = response.json()
                labels = [max(res if isinstance(res, list) else [res], key=lambda x: x['score'])['label'].lower() for res in results]
            elif response.status_code == 503:
                st.warning("⏳ The AI is waking up on the cloud server. Please wait 20 seconds and refresh.")
        except Exception as e:
            st.warning(f"⚠️ Network Error (Safely bypassed): {e}")

# Build Data Dictionary
data = []
for entry, label in zip(valid_news, labels):
    dt = datetime(*entry.published_parsed[:6]).date()
    score = 1 if label == "positive" else -1 if label == "negative" else 0
    data.append({"date": dt, "label": label, "score": score, "title": entry.title, "link": getattr(entry, 'link', '#')})

df = pd.DataFrame(data)
if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

# =========================
# 4. BUILD UI
# =========================
# Metrics
st.markdown("### 📊 Sentiment Overview")
cols = st.columns(3)
cols[0].metric("🟢 Positive News", len(df[df['label'] == 'positive']))
cols[1].metric("🔴 Negative News", len(df[df['label'] == 'negative']))
cols[2].metric("⚪ Neutral News", len(df[df['label'] == 'neutral']))

# Charts
df_daily = df.groupby("date")["score"].mean().rolling(2, min_periods=1).mean()
close_col = price["Close"]
if isinstance(close_col, pd.DataFrame):
    close_col = close_col.iloc[:, 0]
price_daily = close_col.groupby(price.index.date).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily.values, mode='lines+markers', name="Stock Price", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily.values, mode='lines+markers', name="Sentiment Score", yaxis="y2", line=dict(color='orange')))
fig.update_layout(title=f"{ticker} Price vs Sentiment Trend", yaxis2=dict(overlaying="y", side="right", showgrid=False), height=400, margin=dict(l=0, r=0, t=40, b=0))
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

# =========================
# 5. AUTO-REFRESH LOGIC
# =========================
st.divider()
with st.spinner("Dashboard is live. Next automated refresh in 5 minutes..."):
    time.sleep(300) 
    st.rerun()
