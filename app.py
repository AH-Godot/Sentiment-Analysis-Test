# =========================
# ✅ 关键：防崩溃设置（必须最前面）
# =========================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_VISUAL_BACKENDS"] = "1"

import torch
torch.set_num_threads(1)

# =========================
# 导入
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

# =========================
# 页面
# =========================
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker", "AAPL").upper()

# =========================
# ✅ 模型加载（轻量 + 容错）
# =========================
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

    summarizer = None
    try:
        summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )
    except:
        pass  # fallback later

    return sentiment_model, summarizer

sentiment_model, summarizer = load_models()

# =========================
# 数据
# =========================
@st.cache_data(ttl=600)
def get_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(url).entries

@st.cache_data(ttl=600)
def get_price(ticker):
    return yf.download(ticker, period="7d", interval="1h")

news = get_news(ticker)
price = get_price(ticker)

# =========================
# 情感分析
# =========================
week_ago = datetime.now().date() - timedelta(days=7)

data = []
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
    st.warning("No data available")
    st.stop()

# =========================
# 时间序列
# =========================
df_daily = df.groupby("date")["score"].mean().rolling(2).mean()

price["date"] = price.index.date
price_daily = price.groupby("date")["Close"].mean()

# =========================
# ✅ Trading Signal
# =========================
def generate_signal(df_daily, price_daily):
    merged = pd.merge(price_daily, df_daily, left_index=True, right_index=True).dropna()

    if len(merged) < 3:
        return "HOLD", 0

    corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
    trend = df_daily.tail(3).mean()

    if trend > 0.3 and corr > 0.2:
        return "BUY", corr
    elif trend < -0.3 and corr > 0.2:
        return "SELL", corr
    else:
        return "HOLD", corr

signal, corr = generate_signal(df_daily, price_daily)

st.markdown("## 🚦 Trading Signal")

if signal == "BUY":
    st.success(f"🟢 BUY | corr={corr:.2f}")
elif signal == "SELL":
    st.error(f"🔴 SELL | corr={corr:.2f}")
else:
    st.info(f"⚪ HOLD | corr={corr:.2f}")

# =========================
# Dashboard
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    counts = df["label"].value_counts()
    fig_sent = go.Figure()
    fig_sent.add_bar(
        x=counts.index,
        y=counts.values,
        marker_color=["green" if x=="positive" else "red" for x in counts.index]
    )
    fig_sent.update_layout(height=250)
    st.plotly_chart(fig_sent, use_container_width=True)

with col2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price_daily.index,
        y=price_daily.values,
        name="Price",
        line=dict(color="blue", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_daily.index,
        y=df_daily.values,
        name="Sentiment",
        yaxis="y2",
        line=dict(color="red", width=3)
    ))

    fig.update_layout(
        height=300,
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Sentiment", overlaying="y", side="right"),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# ✅ TF-IDF 关键词
# =========================
def extract_keywords(texts, top_k=10):
    vec = TfidfVectorizer(stop_words="english", max_features=50)
    X = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    words = vec.get_feature_names_out()

    return sorted(zip(words, scores), key=lambda x: x[1], reverse=True)[:top_k]

keywords = extract_keywords(df["title"].tolist())

# =========================
# ✅ 关键词情绪
# =========================
keyword_sentiment = {}

for _, row in df.iterrows():
    for word, _ in keywords:
        if word in row["title"].lower():
            keyword_sentiment.setdefault(word, []).append(row["score"])

keyword_sentiment = {
    k: sum(v)/len(v) for k, v in keyword_sentiment.items()
}

# =========================
# ✅ 可视化：词云 + 情绪
# =========================
st.markdown("### 🔑 Market Themes")

colA, colB = st.columns(2)

with colA:
    def get_color(word):
        val = keyword_sentiment.get(word, 0)
        return "green" if val > 0.2 else "red" if val < -0.2 else "gray"

    wc = WordCloud(width=500, height=250, background_color="white")
    wc.generate_from_frequencies(dict(keywords))
    wc.recolor(color_func=lambda word, *args, **kwargs: get_color(word))

    fig_wc, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig_wc)

with colB:
    ks_df = pd.DataFrame(keyword_sentiment.items(), columns=["kw","sent"])

    fig2 = go.Figure()
    fig2.add_bar(
        x=ks_df["kw"],
        y=ks_df["sent"],
        marker=dict(color=ks_df["sent"], colorscale=["red","gray","green"])
    )
    fig2.update_layout(height=250)
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# ✅ AI Summary（稳定版）
# =========================
st.markdown("### 🧠 AI Summary")

def generate_summary(titles, keywords):
    if summarizer:
        try:
            text = " ".join(titles[:5])[:300]

            return summarizer(
                f"summarize: {text}",
                max_length=50
            )[0]["generated_text"]
        except:
            pass

    # fallback
    main_kw = ", ".join([k for k, _ in keywords[:3]])
    return f"Market news focuses on {main_kw} with mixed sentiment."

with st.spinner("Generating summary..."):
    st.success(generate_summary(df["title"].tolist(), keywords))

# =========================
# ✅ 投资建议
# =========================
st.markdown("### 💡 Investment Advice")

def generate_advice(signal, trend, corr, keyword_sentiment):

    top_kw = sorted(keyword_sentiment.items(),
                    key=lambda x: abs(x[1]), reverse=True)[:3]

    kws = [k for k, _ in top_kw]

    if signal == "BUY":
        return f"""
📈 **BUY**

- Positive sentiment ({trend:.2f})
- Strong correlation ({corr:.2f})
- Drivers: {', '.join(kws)}

➡️ Strategy: short-term long
⚠️ Risk: sentiment reversal
"""
    elif signal == "SELL":
        return f"""
📉 **SELL**

- Negative sentiment ({trend:.2f})
- News pressure
- Risks: {', '.join(kws)}

➡️ Strategy: reduce exposure
⚠️ Risk: rebound
"""
    else:
        return f"""
📊 **HOLD**

- Mixed signals ({trend:.2f})
- Weak correlation ({corr:.2f})

➡️ Strategy: wait
⚠️ Risk: breakout
"""

trend = df_daily.tail(3).mean()
st.markdown(generate_advice(signal, trend, corr, keyword_sentiment))

# =========================
# 新闻
# =========================
st.markdown("### 📰 News")

for entry in news[:10]:
    st.write(f"• {entry.title}")
    st.write(entry.link)
    st.write("---")
