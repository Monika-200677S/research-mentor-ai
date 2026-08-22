import re

from nltk.stem import WordNetLemmatizer


# Common English stopwords.
# Stored locally so the preprocessing pipeline does not
# depend on downloading NLTK data at runtime.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been",
    "being", "but", "by", "for", "from", "had", "has",
    "have", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves"
}


lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> list[str]:
    """
    Classical NLP preprocessing.

    Steps:
    1. Lowercase
    2. Tokenization
    3. Stopword removal
    4. Lemmatization
    """

    if not text:
        return []

    # 1. Lowercase
    text = text.lower()

    # 2. Tokenization
    tokens = re.findall(r"\b[a-z]+\b", text)

    # 3. Stopword removal
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    # 4. Lemmatization
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    return tokens