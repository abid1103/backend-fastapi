# backend/reddit_api.py
import praw
import os
import datetime as dt
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


def get_reddit_posts(keyword: str, limit=100):
    """Fetch posts with keyword in all subreddits"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            user_agent=os.getenv("USER_AGENT") or "reddit-sentiment-app",
            check_for_async=False
        )

        posts = []
        for sub in reddit.subreddit("all").search(keyword, sort="new", limit=limit):
            created_dt = dt.datetime.fromtimestamp(
                sub.created_utc, tz=dt.timezone.utc)

            posts.append({
                "reddit_id": sub.id,  # <--- add this
                "title": sub.title or "",
                "score": int(sub.score or 0),
                "url": f"https://www.reddit.com{sub.permalink}",
                "created_utc": created_dt,
                "num_comments": int(sub.num_comments or 0)
            })

        return posts
    except Exception as e:
        print("Reddit fetch error:", e)
        return []


def get_comments_for_post(post_id: str):
    """Fetch all top-level comments for a Reddit post"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            user_agent=os.getenv("USER_AGENT") or "reddit-sentiment-app",
            check_for_async=False
        )
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        comments = [c.body for c in submission.comments.list()
                    if is_english(c.body)]
        return comments
    except Exception as e:
        print(f"Error fetching comments for post {post_id}: {e}")
        return []


def is_english(text: str) -> bool:
    """Detect English using langdetect"""
    try:
        return detect(text) == "en"
    except:
        return False
