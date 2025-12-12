# utils/category_model.py

from transformers import pipeline

# Initialize zero-shot classification pipeline
# This will automatically download a pre-trained model (bart-large-mnli)
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# Define your categories
CATEGORIES = ["Sports", "Stocks", "Crypto",
              "Tech", "Politics", "Entertainment"]

CONFIDENCE_THRESHOLD = 0.6  # below this, fallback to "other"


def predict_category(keyword: str) -> str:
    """
    Predicts the category for a given keyword.
    Returns one of the categories or "other".
    """
    if not keyword:
        return "other"

    result = classifier(keyword, candidate_labels=CATEGORIES)
    top_score = result["scores"][0]
    top_label = result["labels"][0]

    if top_score >= CONFIDENCE_THRESHOLD:
        return top_label
    return "other"
