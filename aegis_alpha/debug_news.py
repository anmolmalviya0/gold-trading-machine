import yfinance as yf
import json
from datetime import datetime

def test_news():
    print(f"--- DEBUG NEWS PULSE [{datetime.now().isoformat()}] ---")
    ticker = yf.Ticker("BTC-USD")
    news = ticker.news
    print(f"Total items fetched: {len(news)}")
    
    for i, n in enumerate(news[:5]):
        content = n.get('content', n)
        title = content.get('title', n.get('title'))
        pub_date = content.get('pubDate', n.get('pubDate'))
        print(f"[{i}] {pub_date} | {title}")

if __name__ == "__main__":
    test_news()
