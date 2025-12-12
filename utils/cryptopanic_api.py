import os
import requests
from datetime import datetime

CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY")

if not CRYPTOPANIC_KEY:
    raise Exception("CRYPTOPANIC_KEY is not set!")


def fetch_cryptopanic_posts(keyword: str):
    url = f"https://cryptopanic.com/api/developer/v2/posts/?auth_token={CRYPTOPANIC_KEY}&currencies={keyword}&public=true"
    print(f"Fetching Cryptopanic posts for: {keyword}")

    response = requests.get(url)
    if response.status_code != 200:
        print("Cryptopanic API error:", response.text)
        return []

    data = response.json()
    posts = []

    for item in data.get("results", []):
        published = item.get("published_at")
        try:
            # handle both with/without milliseconds
            published_dt = datetime.strptime(
                published, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            published_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")

        posts.append({
            "source": "cryptopanic",
            "title": item.get("title"),
            "content": item.get("body") or "",
            "url": item.get("url"),
            "published_at": published_dt
        })

    print(f"Found {len(posts)} posts from Cryptopanic")
    return posts
