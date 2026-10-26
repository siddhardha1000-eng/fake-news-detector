"""
Text Preprocessing Module for Fake News Detection
Covers:
- Text normalization & cleaning (HTML, URLs, punctuation, casing)
- Contraction expansion
- Tokenization (NLTK / Regex fallback)
- Stopword removal & WordNet Lemmatization
- TF-IDF Vectorization with unigrams & bigrams (n-grams)
- Feature extraction & keyword explainability
"""

import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Pre-compile regex patterns for high-throughput cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Common English contractions dictionary
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "'ve": " have",
    "'re": " are",
    "'d": " would",
    "'ll": " will"
}

# Initialize stopwords and lemmatizer safely with fallback
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "after"
        }

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    nltk.download("wordnet", quiet=True)
    LEMMATIZER = WordNetLemmatizer()


def expand_contractions(text: str) -> str:
    """Expands standard English contractions."""
    lower_text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        lower_text = lower_text.replace(contraction, expansion)
    return lower_text


def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    1. Converting to lowercase
    2. Stripping URLs and web links
    3. Stripping HTML tags
    4. Expanding contractions
    5. Removing non-alphabetic characters and punctuation
    6. Collapsing excessive whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = expand_contractions(text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenizes text, strips stopwords, and applies WordNet lemmatization.
    Falls back gracefully to regex splitting if NLTK punkt is unavailable.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    result_tokens = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if remove_stopwords and token in STOP_WORDS:
            continue
        try:
            lemma = LEMMATIZER.lemmatize(token)
        except Exception:
            lemma = token
        result_tokens.append(lemma)

    return result_tokens


def preprocess_text(text: str) -> str:
    """
    Full text preprocessing pipeline for a single string.
    Returns cleaned, lemmatized, space-separated string suitable for TF-IDF.
    """
    tokens = tokenize_and_lemmatize(text)
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", title_col: Optional[str] = "title") -> pd.Series:
    """
    Preprocesses a pandas DataFrame containing news articles.
    If title_col is provided, combines title and text for stronger signal.
    """
    if title_col and title_col in df.columns:
        combined_text = df[title_col].fillna("") + " " + df[text_col].fillna("")
    else:
        combined_text = df[text_col].fillna("")

    return combined_text.apply(preprocess_text)


def create_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Configures and instantiates TF-IDF Vectorizer with n-grams support.

    Args:
        ngram_range: Tuple (min_n, max_n). Default (1, 2) includes unigrams and bigrams.
        max_features: Maximum vocabulary size.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency ratio to filter out overly frequent terms.
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).

    Returns:
        Configured scikit-learn TfidfVectorizer.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode"
    )


def get_top_features_for_text(
    text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Extracts the highest-scoring TF-IDF n-grams from an input text.
    Provides explainability for model predictions.
    """
    processed = preprocess_text(text)
    if not processed:
        return []

    tfidf_vec = vectorizer.transform([processed])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = tfidf_vec.toarray()[0]

    top_indices = scores.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((feature_names[idx], round(float(scores[idx]), 4)))
    return results


if __name__ == "__main__":
    sample = "BREAKING: Scientists have announced that the James Webb Space Telescope spotted carbon dioxide on exoplanet WASP-39b!"
    print("Raw Sample:", sample)
    print("Cleaned:", clean_text(sample))
    print("Tokens/Lemmas:", tokenize_and_lemmatize(sample))
    print("Preprocessed String:", preprocess_text(sample))
