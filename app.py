import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime

# 页面
st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide")
st.title("📈 美股情感 + 股价联动分析")

ticker = st.text_input("输入股票代码", "AAPL").upper()

# ✅ 加载模型（缓存）
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

model = load_model()

# ✅ 获取新闻
@st.cache_data(ttl=600)
def get_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(url).entries

# ✅ 获取股价
@st.cache_data(ttl=600)
def get_price(ticker):
    data = yf.download(ticker, period="7d", interval="1h")
    return data

# -------------------
# 获取数据
# -------------------
news = get_news(ticker)
price = get_price(ticker)

# -------------------
# 情感分析
# -------------------
sentiment_data = []

for entry in news[:30]:
    title = entry.title
    
    # 时间处理
    if hasattr(entry, "published_parsed"):
        date = datetime(*entry.published_parsed[:6])
    else:
        date = datetime.today()

    result = model(title)[0]
    label = result["label"]

    sentiment_score = 0
    if label == "positive":
        sentiment_score = 1
    elif label == "negative":
        sentiment_score = -1

    sentiment_data.append({
        "date": date.date(),
        "score": sentiment_score
    })

# 转 DataFrame
df_sentiment = pd.DataFrame(sentiment_data)

if len(df_sentiment) > 0:
    df_sentiment = df_sentiment.groupby("date").mean()
else:
    st.warning("没有情感数据")

# -------------------
# 股价处理
# -------------------
price["date"] = price.index.date
price_daily = price.groupby("date")["Close"].mean()

# -------------------
# 联动图
# -------------------
st.subheader("📊 股价 + 情感联动图")

fig, ax1 = plt.subplots(figsize=(10, 5))

# 股价（左轴）
ax1.plot(price_daily.index, price_daily.values)
ax1.set_ylabel("Stock Price")

# 情感（右轴）
ax2 = ax1.twinx()

if len(df_sentiment) > 0:
    ax2.plot(df_sentiment.index, df_sentiment["score"])
    ax2.set_ylabel("Sentiment (-1 ~ 1)")

plt.title(f"{ticker} Price vs Sentiment")

st.pyplot(fig)

# -------------------
# 分析总结
# -------------------
st.subheader("📌 分析总结")

if len(df_sentiment) > 0:
    avg_sentiment = df_sentiment["score"].mean()

    if avg_sentiment > 0.2:
        st.success("市场情绪：偏乐观 🟢")
    elif avg_sentiment < -0.2:
        st.error("市场情绪：偏悲观 🔴")
    else:
        st.info("市场情绪：中性 ⚪")

# -------------------
# 明细
# -------------------
st.subheader("📰 新闻详情")

for entry in news[:10]:
    st.markdown(f"**{entry.title}**")
    st.write(entry.link)
    st.write("---")
