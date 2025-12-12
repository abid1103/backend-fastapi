import os
import requests
from datetime import datetime

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


def fetch_newsapi_posts(keyword: str):
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={NEWSAPI_KEY}&pageSize=10"
    response = requests.get(url).json()
    posts = []
    for article in response.get("articles", []):
        posts.append({
            "source": "newsapi",
            "title": article.get("title"),
            "content": article.get("content") or article.get("description") or "",
            "url": article.get("url"),
            "published_at": datetime.strptime(article.get("publishedAt"), "%Y-%m-%dT%H:%M:%SZ")
        })
    return posts
