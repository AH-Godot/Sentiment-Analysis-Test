import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter

# 页面
st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide")
st.title("📈 美股情感分析 + 股价联动")

ticker = st.text_input("输入股票代码", "AAPL").upper()

# -------------------------
# 模型加载
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

model = load_model()

# -------------------------
# 数据获取
# -------------------------
@st.cache_data(ttl=600)
def get_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(url).entries

@st.cache_data(ttl=600)
def get_price(ticker):
    return yf.download(ticker, period="7d", interval="1h")

news = get_news(ticker)
price = get_price(ticker)

# -------------------------
# 情感分析 + 最近一周
# -------------------------
sentiment_data = []
week_ago = datetime.now().date() - timedelta(days=7)

for entry in news[:50]:
    title = entry.title

    if hasattr(entry, "published_parsed"):
        date = datetime(*entry.published_parsed[:6]).date()
    else:
        continue

    if date < week_ago:
        continue

    result = model(title)[0]
    label = result["label"]

    score = 0
    if label == "positive":
        score = 1
    elif label == "negative":
        score = -1

    sentiment_data.append({
        "date": date,
        "label": label,
        "score": score,
        "title": title
    })

df = pd.DataFrame(sentiment_data)

# =========================
# ✅ ⭐ 一周情绪总览（核心）
# =========================
st.header("🧠 最近一周市场情绪总览")

if len(df) > 0:

    # 情感统计
    counts = Counter(df["label"])

    # ✅ 图1：情感分布
    fig1, ax1 = plt.subplots()
    ax1.bar(counts.keys(), counts.values())
    ax1.set_title("Sentiment Distribution (7 Days)")

    st.pyplot(fig1)

    # ✅ 情绪结论
    avg_score = df["score"].mean()

    if avg_score > 0.2:
        sentiment_summary = "整体偏乐观 🟢"
        st.success(f"📌 结论：{sentiment_summary}")
    elif avg_score < -0.2:
        sentiment_summary = "整体偏悲观 🔴"
        st.error(f"📌 结论：{sentiment_summary}")
    else:
        sentiment_summary = "整体偏中性 ⚪"
        st.info(f"📌 结论：{sentiment_summary}")

    # =========================
    # ✅ ⭐ 新闻总结（自动）
    # =========================
    st.subheader("📰 本周新闻总结")

    # 按情绪挑选代表新闻
    top_positive = df[df["score"] == 1]["title"].head(3).tolist()
    top_negative = df[df["score"] == -1]["title"].head(3).tolist()

    summary_text = "### 🔎 市场关键信号\n"

    if top_positive:
        summary_text += "\n🟢 利好消息：\n"
        for t in top_positive:
            summary_text += f"- {t}\n"

    if top_negative:
        summary_text += "\n🔴 利空消息：\n"
        for t in top_negative:
            summary_text += f"- {t}\n"

    summary_text += f"\n📊 本周整体情绪：{sentiment_summary}"

    st.markdown(summary_text)

else:
    st.warning("本周没有足够新闻数据")

# =========================
# ✅ 股价 + 情感联动
# =========================
st.header("📊 股价 + 情感联动分析")

if len(df) > 0:

    df_daily = df.groupby("date")["score"].mean()

    price["date"] = price.index.date
    price_daily = price.groupby("date")["Close"].mean()

    fig2, ax1 = plt.subplots(figsize=(10, 5))

    # 股价
    ax1.plot(price_daily.index, price_daily.values)
    ax1.set_ylabel("Stock Price")

    # 情感
    ax2 = ax1.twinx()
    ax2.plot(df_daily.index, df_daily.values)
    ax2.set_ylabel("Sentiment")

    plt.title(f"{ticker} Price vs Sentiment")

    st.pyplot(fig2)

# =========================
# ✅ 新闻列表
# =========================
st.header("📰 新闻详情")

for entry in news[:10]:
    st.markdown(f"**{entry.title}**")
    st.write(entry.link)
    st.write("---")
