from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize analyzer once
sia = SentimentIntensityAnalyzer()


def analyze_sentiment(texts):
    """
    Takes a string or list of strings and returns sentiment scores for each,
    along with an overall summary (positive vs negative only, neutral removed).
    """
    if isinstance(texts, str):  # handle single string input
        texts = [texts]

    results = []
    overall = {"positive": 0, "negative": 0}

    for text in texts:
        score = sia.polarity_scores(text)
        sentiment = None

        if score["compound"] >= 0.05:
            sentiment = "positive"
            overall["positive"] += 1
        elif score["compound"] <= -0.05:
            sentiment = "negative"
            overall["negative"] += 1
        else:
            # skip neutral
            continue

        results.append({
            "text": text,
            "sentiment": sentiment,
            "score": score
        })

    return {
        "detailed": results,
        "summary": overall
    }
