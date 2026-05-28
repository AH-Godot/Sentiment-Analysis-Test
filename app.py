import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# TIP: Add your Hugging Face Token in Streamlit Cloud Secrets (Settings -> Secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =========================
# Core Data Fetching
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker_symbol):
    url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        return feedparser.parse(url).entries[:15] # Limit to 15 to keep API calls fast
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_price(ticker_symbol):
    try:
        return yf.download(ticker_symbol, period="7d", interval="1h")
    except Exception:
        return pd.DataFrame()

with st.spinner("Fetching market data..."):
    news = get_news(ticker)
    price = get_price(ticker)

if not news or price.empty:
    st.warning("Data unavailable for this ticker. Please try another symbol.")
    st.stop()

# =========================
# Pipeline 1: Sentiment Analysis via API
# =========================
def get_sentiment(text):
    # NOTE: Replace this URL with YOUR fine-tuned model URL once you upload it to Hugging Face!
    API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": text}, timeout=10)
        if response.status_code == 200:
            results = response.json()[0]
            best = max(results, key=lambda x: x['score'])
            return best['label']
    except Exception:
        pass
    return "neutral" # Fallback if API fails

data = []
with st.spinner("Analyzing Sentiment..."):
    week_ago = datetime.now().date() - timedelta(days=7)
    
    for entry in news:
        if not hasattr(entry, "published_parsed"): continue
        date = datetime(*entry.published_parsed[:6]).date()
        if date < week_ago: continue
            
        label = get_sentiment(entry.title)
        score = 1 if label == "positive" else -1 if label == "negative" else 0
        
        # ✅ Added 'link' to the data dictionary
        data.append({
            "date": date, 
            "label": label, 
            "score": score, 
            "title": entry.title,
            "link": entry.link 
        })

df = pd.DataFrame(data)
if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

# =========================
# ✅ Sentiment Counting Indicator
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
# Charts & Visuals (MultiIndex Fix applied)
# =========================
df_daily = df.groupby("date")["score"].mean().rolling(2, min_periods=1).mean()

close_col = price["Close"]
if isinstance(close_col, pd.DataFrame):
    close_col = close_col.iloc[:, 0]

price_daily = close_col.groupby(price.index.date).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=price_daily.index, 
    y=price_daily.values, 
    mode='lines+markers',
    name="Stock Price", 
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=df_daily.index, 
    y=df_daily.values, 
    mode='lines+markers', 
    name="Sentiment Score", 
    yaxis="y2", 
    line=dict(color='orange')
))

fig.update_layout(
    title=f"{ticker} Price vs Sentiment Trend",
    yaxis2=dict(overlaying="y", side="right", showgrid=False),
    height=400,
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Pipeline 2: Market Summary via API
# =========================
st.markdown("### 🧠 AI Market Summary")

def get_summary(news_titles, sentiment_trend):
    API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
    prompt = f"The stock sentiment is currently {sentiment_trend}. Based on news like '{news_titles[0]}', the market outlook is: "
    
    try:
        response = requests.post(
            API_URL, 
            headers=HEADERS, 
            json={"inputs": prompt, "parameters": {"max_new_tokens": 30}}, 
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()[0]["generated_text"]
            return result.replace(prompt, "").split('.')[0] + "." 
    except Exception:
        pass
    
    return f"The market remains actively traded with a {sentiment_trend} tilt based on recent headlines."

with st.spinner("Generating Summary..."):
    trend = "positive" if df_daily.iloc[-1] > 0 else "negative" if df_daily.iloc[-1] < 0 else "neutral"
    titles = df["title"].tolist()
    summary = get_summary(titles, trend)
    
st.success(summary)

# =========================
# ✅ News Feed (Now with clickable links)
# =========================
st.markdown("### 📰 Recent Headlines")
for _, row in df.head(10).iterrows():
    emoji = "🟢" if row["label"] == "positive" else "🔴" if row["label"] == "negative" else "⚪"
    # Converts the title into a clickable markdown hyperlink
    st.markdown(f"{emoji} [{row['title']}]({row['link']})")
