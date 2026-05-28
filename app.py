# =========================
# ✅ 防崩设置（必须最前）
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
import re

# =========================
# 页面
# =========================
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📈 AI Stock Sentiment Dashboard")

ticker = st.text_input("Ticker", "AAPL").upper()

# =========================
# ✅ 两个 pipeline
# =========================
@st.cache_resource
def load_models():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

    generator = None
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2"
        )
    except:
        pass

    return sentiment_model, generator

sentiment_model, generator = load_models()

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
# 图表
# =========================
col1, col2 = st.columns([1,2])

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
    fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily, name="Price"))
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily, name="Sentiment", yaxis="y2"))

    fig.update_layout(
        height=300,
        yaxis2=dict(overlaying="y", side="right")
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# ✅ TF-IDF关键词
# =========================
def extract_keywords(texts):
    vec = TfidfVectorizer(stop_words="english", max_features=50)
    X = vec.fit_transform(texts)

    scores = X.sum(axis=0).A1
    words = vec.get_feature_names_out()

    return sorted(zip(words, scores), key=lambda x: x[1], reverse=True)[:10]

keywords = extract_keywords(df["title"].tolist())

# =========================
# ✅ 关键词情绪
# =========================
keyword_sentiment = {}

for _, row in df.iterrows():
    for word, _ in keywords:
        if word in row["title"].lower():
            keyword_sentiment.setdefault(word, []).append(row["score"])

keyword_sentiment = {k: sum(v)/len(v) for k, v in keyword_sentiment.items()}

# =========================
# ✅ 词云
# =========================
st.markdown("### 🔑 Market Themes")

colA, colB = st.columns(2)

with colA:
    def get_color(word):
        val = keyword_sentiment.get(word, 0)
        return "green" if val > 0 else "red" if val < 0 else "gray"

    wc = WordCloud(width=500, height=250)
    wc.generate_from_frequencies(dict(keywords))
    wc.recolor(color_func=lambda word, *args, **kwargs: get_color(word))

    fig, ax = plt.subplots(figsize=(5,3))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

with colB:
    ks_df = pd.DataFrame(keyword_sentiment.items(), columns=["kw","sent"])
    st.bar_chart(ks_df.set_index("kw"))

# =========================
# ✅ Summary（高级版）
# =========================
st.markdown("### 🧠 Market Summary")

def generate_summary(titles, keywords, keyword_sentiment, signal):

    top_kw = [k for k, _ in keywords[:3]]
    pos_kw = [k for k, v in keyword_sentiment.items() if v > 0]
    neg_kw = [k for k, v in keyword_sentiment.items() if v < 0]

    keyword_hint = ", ".join(top_kw)

    tone = "positive" if signal == "BUY" else "negative" if signal == "SELL" else "neutral"

    input_text = " ".join(titles[:5])[:300]

    prompt = f"""
You are a financial analyst.

Write ONE professional sentence summarizing market sentiment.
Tone: {tone}
Focus on: {keyword_hint}
Include both opportunity and risk.

News:
{input_text}

Answer:
"""

    if generator:
        try:
            result = generator(prompt, max_new_tokens=40, do_sample=False)[0]["generated_text"]
            text = result.replace(prompt, "").strip()

            match = re.search(r"(.+?\.)", text)
            if match:
                sentence = match.group(1)
            else:
                sentence = text[:200]

        except:
            sentence = None
    else:
        sentence = None

    if not sentence:
        pos = ", ".join(pos_kw[:2]) or "limited upside"
        neg = ", ".join(neg_kw[:2]) or "limited downside"

        if signal == "BUY":
            sentence = f"Market sentiment is positive, supported by {pos}, though risks remain around {neg}."
        elif signal == "SELL":
            sentence = f"Market sentiment is negative, driven by {neg}, with limited support from {pos}."
        else:
            sentence = f"Market sentiment is mixed, with support from {pos} offset by risks in {neg}."

    extra = f"""
✅ **Opportunities:** {', '.join(pos_kw[:3]) or 'None'}  
⚠️ **Risks:** {', '.join(neg_kw[:3]) or 'None'}
"""

    return sentence, extra


summary, extra = generate_summary(
    df["title"].tolist(),
    keywords,
    keyword_sentiment,
    signal
)

st.success(summary)
st.markdown(extra)

# =========================
# 新闻
# =========================
st.markdown("### 📰 News")

for entry in news[:10]:
    st.write(f"• {entry.title}")
    st.write(entry.link)
    st.write("---")
