"""
helpers.py
----------
Utility functions for data loading, preprocessing, and model initialization.

Data source priority:
  1. Kaggle processed data  (backend/data/kaggle/processed_*.csv)  ← real data
  2. Built-in sample data   (backend/data/sample_data.py)          ← fallback
"""

import logging

logger = logging.getLogger(__name__)

_collaborative_model = None
_content_model = None
_data_source = None   # "kaggle" | "sample"


def _load_data():
    """
    Load movies + ratings, preferring Kaggle processed data over sample data.
    Returns (movies_df, ratings_df, source_name)
    """
    try:
        from backend.data.kaggle_loader import load_processed, kaggle_data_available
        if kaggle_data_available():
            movies_df, ratings_df = load_processed()
            logger.info(f"✅ Loaded Kaggle data — {len(movies_df):,} movies, {len(ratings_df):,} ratings")
            return movies_df, ratings_df, "kaggle"
    except Exception as e:
        logger.warning(f"Kaggle data unavailable ({e}), falling back to sample data.")

    from backend.data.sample_data import get_movies_df, get_ratings_df
    movies_df  = get_movies_df()
    ratings_df = get_ratings_df()
    logger.info(f"ℹ️  Using built-in sample data — {len(movies_df)} movies, {len(ratings_df)} ratings")
    return movies_df, ratings_df, "sample"


def get_trained_models():
    """
    Initialize and return cached (collaborative, content_based) models.
    Models are trained once on first call and reused thereafter.
    """
    global _collaborative_model, _content_model, _data_source

    if _collaborative_model is None or _content_model is None:
        from backend.models.collaborative import CollaborativeFilter
        from backend.models.content_based import ContentBasedFilter

        movies_df, ratings_df, source = _load_data()
        _data_source = source

        _collaborative_model = CollaborativeFilter()
        _collaborative_model.fit(ratings_df, movies_df)

        _content_model = ContentBasedFilter()
        _content_model.fit(movies_df)

    return _collaborative_model, _content_model


def get_data_source() -> str:
    """Return which data source is active ('kaggle' or 'sample')."""
    if _data_source is None:
        get_trained_models()
    return _data_source


def get_movies_and_ratings():
    """Return raw (movies_df, ratings_df) from whichever source is active."""
    return _load_data()[:2]


def serialize_movie(movie: dict) -> dict:
    """Ensure all values in a movie dict are JSON-serializable."""
    return {
        "movie_id":        int(movie.get("movie_id", 0)),
        "title":           str(movie.get("title", "")),
        "genres":          list(movie.get("genres", [])),
        "year":            int(movie.get("year", 0)),
        "predicted_rating": float(movie["predicted_rating"]) if "predicted_rating" in movie else None,
        "similarity_score": float(movie["similarity_score"]) if "similarity_score" in movie else None,
    }
