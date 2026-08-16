"""
predictor.py
=============================================================
Shared prediction logic for the Bookly web application.

This module loads the trained model and the encoding maps once,
then exposes a single predict_rating() function. Keeping this
logic separate from the Streamlit UI means the exact same
transformation path can be reused for single books and for
batch CSV uploads, with no risk of the two drifting apart.

The transformation MUST mirror Step 3 (feature engineering)
exactly, because the model was trained on features built that
way. If the app encoded inputs differently from training, the
model would receive values it never saw and return nonsense.
=============================================================
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------
# Locate model artifacts relative to this file, so the app works
# no matter which directory Streamlit is launched from.
# ---------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).parent

_model = joblib.load(ARTIFACT_DIR / "bookly_model.pkl")
_publisher_encoding = joblib.load(ARTIFACT_DIR / "publisher_encoding.pkl")
_author_encoding = joblib.load(ARTIFACT_DIR / "author_encoding.pkl")
_global_mean = joblib.load(ARTIFACT_DIR / "global_mean.pkl")

with open(ARTIFACT_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

# Rating scale bounds. Predictions are clipped to this range so
# the app never shows an impossible value like 5.3 or 0.8.
RATING_MIN = 1.0
RATING_MAX = 5.0


def encode_publisher(publisher_name):
    """
    Convert a publisher name into its trained numeric encoding.

    During training we replaced each publisher with the smoothed
    mean rating of that publisher's books (target encoding). Here
    we look that value up. A publisher the model never saw during
    training falls back to the global mean, which is the neutral,
    no-information guess.
    """
    return _publisher_encoding.get(publisher_name, _global_mean)


def encode_authors(authors_string):
    """
    Convert an author string into a single numeric encoding.

    Books can have several authors separated by '/'. We look up
    each author's trained encoding and average them, which is the
    same rule used during training. Unknown authors fall back to
    the global mean.
    """
    names = [name.strip() for name in str(authors_string).split("/")]
    values = [_author_encoding.get(name, _global_mean) for name in names]
    return float(np.mean(values))


def build_feature_row(num_pages, ratings_count, text_reviews_count,
                      pub_year, series_num, num_authors, title_words,
                      publisher, authors):
    """
    Assemble one row of model-ready features from raw inputs.

    Every transformation here matches Step 3 feature engineering:
      - ratings_count and text_reviews_count are log-transformed,
        because the raw counts span 0 to millions and would
        otherwise dominate every other feature.
      - publisher and authors are target-encoded via the saved maps.
      - the remaining values pass through as plain numbers.
    """
    row = {
        "num_pages": num_pages,
        "pub_year": pub_year,
        "series_num": series_num,
        "title_words": title_words,
        "num_authors": num_authors,
        "log_ratings_count": np.log1p(ratings_count),
        "log_text_reviews_count": np.log1p(text_reviews_count),
        "publisher_enc": encode_publisher(publisher),
        "author_enc": encode_authors(authors),
    }
    # reindex guarantees the columns are in the exact order the
    # model was trained on. Any missing column is filled with 0.
    return pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)


def predict_rating(num_pages, ratings_count, text_reviews_count,
                   pub_year, series_num, num_authors, title_words,
                   publisher, authors):
    """
    Predict a single book's rating from raw inputs.

    Returns a float in the range 1.0 to 5.0.
    """
    features = build_feature_row(
        num_pages=num_pages,
        ratings_count=ratings_count,
        text_reviews_count=text_reviews_count,
        pub_year=pub_year,
        series_num=series_num,
        num_authors=num_authors,
        title_words=title_words,
        publisher=publisher,
        authors=authors,
    )
    raw = float(_model.predict(features)[0])
    return float(np.clip(raw, RATING_MIN, RATING_MAX))


def predict_batch(df):
    """
    Predict ratings for a whole dataframe of books.

    Expected columns (same names as the original dataset):
      title, authors, language_code, num_pages, ratings_count,
      text_reviews_count, publisher, publication_date

    Any missing column is handled with a sensible default so a
    partially filled CSV still returns predictions. Returns a copy
    of the input with a new 'predicted_rating' column.
    """
    import re

    result = df.copy()

    def row_prediction(row):
        # Derive series_num and title_words from the title if present
        title = str(row.get("title", ""))
        series_match = re.search(r"\(.*?#(\d+)", title)
        series_num = int(series_match.group(1)) if series_match else 0
        title_words = len(title.split()) if title else 3

        authors = str(row.get("authors", "Unknown"))
        num_authors = len(authors.split("/"))

        # Extract year from publication_date if present
        pub_date = str(row.get("publication_date", ""))
        year_match = re.search(r"(\d{4})", pub_date)
        pub_year = int(year_match.group(1)) if year_match else 2000

        return predict_rating(
            num_pages=float(row.get("num_pages", 300) or 300),
            ratings_count=float(row.get("ratings_count", 0) or 0),
            text_reviews_count=float(row.get("text_reviews_count", 0) or 0),
            pub_year=pub_year,
            series_num=series_num,
            num_authors=num_authors,
            title_words=title_words,
            publisher=str(row.get("publisher", "")),
            authors=authors,
        )

    result["predicted_rating"] = result.apply(row_prediction, axis=1).round(2)
    return result


# A short list of well known publishers to offer as quick suggestions
# in the interface. These all exist in the training encoding.
COMMON_PUBLISHERS = [
    "Penguin Books",
    "Penguin Classics",
    "Vintage",
    "HarperCollins",
    "Scholastic Inc.",
    "Ballantine Books",
    "Mariner Books",
    "Bantam",
    "Tor Books",
    "Oxford University Press",
    "VIZ Media",
    "Library of America",
]

# Language options that match the trained categories.
LANGUAGE_OPTIONS = ["eng", "spa", "fre", "ger", "jpn", "zho", "mul", "other"]
