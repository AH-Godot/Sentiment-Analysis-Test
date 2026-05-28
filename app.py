import streamlit as st
import feedparser
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="US Stock Sentiment", layout="wide")

st.title("📈 美股新闻情感分析仪")

# 用户输入
ticker = st.text_input("输入股票代码（如 AAPL, TSLA）", "AAPL")

# 缓存模型（避免重复加载）
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

model = load_model()

# 缓存数据（10分钟）
@st.cache_data(ttl=600)
def fetch_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return feed.entries

# 获取新闻
entries = fetch_news(ticker)

if len(entries) == 0:
    st.warning("没有找到相关新闻")
else:
    results = []

    st.subheader(f"📰 最新新闻（{ticker}）")

    for entry in entries[:15]:
        title = entry.title

        # 情感分析
        sentiment = model(title)[0]
        label = sentiment["label"]
        score = sentiment["score"]

        if label == "positive":
            emoji = "🟢"
        elif label == "negative":
            emoji = "🔴"
        else:
            emoji = "⚪"

        st.markdown(f"### {emoji} {title}")
        st.write(f"情感: {label} | 置信度: {score:.2f}")
        st.write(entry.link)
        st.write("---")

        results.append(label)

    # 情感统计
    st.subheader("📊 情感分布")

    df = pd.Series(results).value_counts()

    fig, ax = plt.subplots()
    df.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

    # 总结
    st.subheader("📌 情绪总结")

    total = len(results)
    pos = results.count("positive")
    neg = results.count("negative")

    sentiment_score = (pos - neg) / total

    if sentiment_score > 0.2:
        st.success("整体情绪：偏乐观 🟢")
    elif sentiment_score < -0.2:
        st.error("整体情绪：偏悲观 🔴")
    else:
        st.info("整体情绪：中性 ⚪")
