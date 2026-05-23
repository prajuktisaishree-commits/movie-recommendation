"""
kaggle_loader.py
----------------
Loads and preprocesses real movie data from Kaggle datasets.

Supported datasets:
  1. MovieLens (grouplens.org) — ratings + movie metadata
     Files needed: movies.csv, ratings.csv, (optional) tags.csv, links.csv
     Kaggle: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
             or the smaller: https://www.kaggle.com/datasets/grouplens/movielens-latest-small

  2. TMDB Movie Metadata (The Movie Database)
     Files needed: movies_metadata.csv, ratings_small.csv, credits.csv, keywords.csv
     Kaggle: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Place the downloaded CSV files inside:
    backend/data/kaggle/

Then run:
    python backend/data/kaggle_loader.py

This script will generate:
    backend/data/kaggle/processed_movies.csv
    backend/data/kaggle/processed_ratings.csv
"""

import os
import ast
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "kaggle")
PROCESSED_MOVIES = os.path.join(KAGGLE_DIR, "processed_movies.csv")
PROCESSED_RATINGS = os.path.join(KAGGLE_DIR, "processed_ratings.csv")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_dataset():
    """
    Auto-detect which Kaggle dataset files are present.
    Returns: 'movielens' | 'tmdb' | None
    """
    files = set(os.listdir(KAGGLE_DIR)) if os.path.isdir(KAGGLE_DIR) else set()
    if "movies_metadata.csv" in files and "ratings_small.csv" in files:
        return "tmdb"
    if "movies.csv" in files and "ratings.csv" in files:
        return "movielens"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MOVIELENS LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_movielens(max_movies: int = 5000, min_ratings: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess MovieLens dataset.

    Parameters
    ----------
    max_movies   : keep only the top-N most-rated movies (controls memory)
    min_ratings  : drop movies with fewer ratings than this

    Returns
    -------
    (movies_df, ratings_df)
    """
    logger.info("Loading MovieLens dataset...")

    movies_path  = os.path.join(KAGGLE_DIR, "movies.csv")
    ratings_path = os.path.join(KAGGLE_DIR, "ratings.csv")
    tags_path    = os.path.join(KAGGLE_DIR, "tags.csv")

    # ── Movies ──────────────────────────────────────────────────────────────
    movies = pd.read_csv(movies_path)
    logger.info(f"  Raw movies: {len(movies):,}")

    # Extract year from title: "Toy Story (1995)" → 1995
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
    movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()

    # Split genres pipe-separated string → list
    movies["genres"] = movies["genres"].apply(
        lambda g: [] if pd.isna(g) or g == "(no genres listed)" else g.split("|")
    )
    movies["genres_str"] = movies["genres"].apply(lambda g: " ".join(g))

    # ── Tags (optional — enriches description) ──────────────────────────────
    if os.path.exists(tags_path):
        tags = pd.read_csv(tags_path)
        tags_agg = (
            tags.groupby("movieId")["tag"]
            .apply(lambda x: " ".join(x.dropna().astype(str).unique()))
            .reset_index()
            .rename(columns={"tag": "tags_text"})
        )
        movies = movies.merge(tags_agg, left_on="movieId", right_on="movieId", how="left")
        movies["tags_text"] = movies["tags_text"].fillna("")
    else:
        movies["tags_text"] = ""

    movies["description"] = movies["genres_str"] + " " + movies["tags_text"]

    # ── Ratings ─────────────────────────────────────────────────────────────
    ratings = pd.read_csv(ratings_path)
    logger.info(f"  Raw ratings: {len(ratings):,}")

    # Filter to most-rated movies for performance
    rating_counts = ratings["movieId"].value_counts()
    popular_ids = rating_counts[rating_counts >= min_ratings].head(max_movies).index
    ratings = ratings[ratings["movieId"].isin(popular_ids)].copy()
    movies  = movies[movies["movieId"].isin(popular_ids)].copy()

    logger.info(f"  After filtering — movies: {len(movies):,}, ratings: {len(ratings):,}")

    # Rename columns to match internal schema
    movies = movies.rename(columns={"movieId": "movie_id"})
    ratings = ratings.rename(columns={"movieId": "movie_id", "userId": "user_id"})

    movies  = movies[["movie_id", "title", "genres", "genres_str", "year", "description"]].reset_index(drop=True)
    ratings = ratings[["user_id", "movie_id", "rating"]].reset_index(drop=True)

    return movies, ratings


# ─────────────────────────────────────────────────────────────────────────────
# TMDB LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _safe_parse(val):
    """Safely parse a stringified Python literal or JSON object."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return []


def load_tmdb(max_movies: int = 5000, min_ratings: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the TMDB / Kaggle 'The Movies Dataset'.

    Parameters
    ----------
    max_movies  : keep top-N most popular movies
    min_ratings : drop movies with fewer ratings

    Returns
    -------
    (movies_df, ratings_df)
    """
    logger.info("Loading TMDB dataset...")

    meta_path    = os.path.join(KAGGLE_DIR, "movies_metadata.csv")
    ratings_path = os.path.join(KAGGLE_DIR, "ratings_small.csv")
    credits_path = os.path.join(KAGGLE_DIR, "credits.csv")
    keywords_path = os.path.join(KAGGLE_DIR, "keywords.csv")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = pd.read_csv(meta_path, low_memory=False)
    logger.info(f"  Raw movies: {len(meta):,}")

    # Drop bad rows (adult films, missing ids)
    meta = meta[meta["adult"] == "False"].copy()
    meta = meta[meta["id"].str.isnumeric().fillna(False)].copy()
    meta["id"] = meta["id"].astype(int)

    # Genres: [{"id":..., "name":"Action"}, ...] → ["Action", ...]
    meta["genres"] = meta["genres"].apply(
        lambda x: [g["name"] for g in _safe_parse(x) if isinstance(g, dict)]
    )
    meta["genres_str"] = meta["genres"].apply(lambda g: " ".join(g))

    # Year from release_date
    meta["year"] = pd.to_datetime(meta["release_date"], errors="coerce").dt.year

    # Keep description from overview
    meta["description"] = meta["overview"].fillna("")

    # Popularity-based filtering
    meta["popularity"] = pd.to_numeric(meta["popularity"], errors="coerce").fillna(0)
    meta = meta.nlargest(max_movies, "popularity")

    # ── Credits (optional) ──────────────────────────────────────────────────
    if os.path.exists(credits_path):
        credits = pd.read_csv(credits_path)
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
        credits = credits.dropna(subset=["id"])
        credits["id"] = credits["id"].astype(int)

        def top_cast(val, n=3):
            actors = _safe_parse(val)
            return " ".join(a["name"] for a in actors[:n] if isinstance(a, dict) and "name" in a)

        def director(val):
            crew = _safe_parse(val)
            dirs = [c["name"] for c in crew if isinstance(c, dict) and c.get("job") == "Director"]
            return dirs[0] if dirs else ""

        credits["cast_text"]     = credits["cast"].apply(top_cast)
        credits["director_text"] = credits["crew"].apply(director)
        meta = meta.merge(credits[["id", "cast_text", "director_text"]], on="id", how="left")
        meta["cast_text"]     = meta["cast_text"].fillna("")
        meta["director_text"] = meta["director_text"].fillna("")
    else:
        meta["cast_text"]     = ""
        meta["director_text"] = ""

    # ── Keywords (optional) ─────────────────────────────────────────────────
    if os.path.exists(keywords_path):
        kw = pd.read_csv(keywords_path)
        kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
        kw = kw.dropna(subset=["id"])
        kw["id"] = kw["id"].astype(int)
        kw["kw_text"] = kw["keywords"].apply(
            lambda x: " ".join(k["name"] for k in _safe_parse(x) if isinstance(k, dict))
        )
        meta = meta.merge(kw[["id", "kw_text"]], on="id", how="left")
        meta["kw_text"] = meta["kw_text"].fillna("")
    else:
        meta["kw_text"] = ""

    # Combine all text features for content model
    meta["description"] = (
        meta["description"] + " "
        + meta["genres_str"] + " "
        + meta["cast_text"] + " "
        + meta["director_text"] + " "
        + meta["kw_text"]
    ).str.strip()

    # ── Ratings ──────────────────────────────────────────────────────────────
    ratings = pd.read_csv(ratings_path)
    logger.info(f"  Raw ratings: {len(ratings):,}")

    # ratings_small uses tmdbId-like ids — we align via id
    ratings = ratings.rename(columns={"movieId": "movie_id", "userId": "user_id"})
    valid_ids = set(meta["id"].tolist())
    ratings = ratings[ratings["movie_id"].isin(valid_ids)]

    # Filter by min_ratings
    counts = ratings["movie_id"].value_counts()
    keep_ids = set(counts[counts >= min_ratings].index)
    ratings = ratings[ratings["movie_id"].isin(keep_ids)]
    meta = meta[meta["id"].isin(keep_ids.union(set(ratings["movie_id"])))]

    logger.info(f"  After filtering — movies: {len(meta):,}, ratings: {len(ratings):,}")

    # Final schema
    meta = meta.rename(columns={"id": "movie_id", "original_title": "title_orig"})
    meta["title"] = meta["title"].fillna(meta.get("title_orig", meta.get("title", "Unknown")))
    movies = meta[["movie_id", "title", "genres", "genres_str", "year", "description"]].reset_index(drop=True)
    ratings = ratings[["user_id", "movie_id", "rating"]].reset_index(drop=True)

    return movies, ratings


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def process_and_save(max_movies: int = 5000):
    """
    Auto-detect dataset, process it, and save to processed_movies.csv
    and processed_ratings.csv.
    """
    dataset = detect_dataset()
    if dataset is None:
        logger.error(
            "No Kaggle dataset files found in backend/data/kaggle/\n"
            "Please download one of the supported datasets. See README for instructions."
        )
        return False

    logger.info(f"Detected dataset: {dataset.upper()}")

    if dataset == "movielens":
        movies, ratings = load_movielens(max_movies=max_movies)
    else:
        movies, ratings = load_tmdb(max_movies=max_movies)

    # Save processed files
    # Convert genres list → JSON string for CSV storage
    movies_out = movies.copy()
    movies_out["genres"] = movies_out["genres"].apply(
        lambda g: json.dumps(g) if isinstance(g, list) else g
    )
    movies_out.to_csv(PROCESSED_MOVIES, index=False)
    ratings.to_csv(PROCESSED_RATINGS, index=False)

    logger.info(f"✅ Saved to:")
    logger.info(f"   {PROCESSED_MOVIES}")
    logger.info(f"   {PROCESSED_RATINGS}")
    logger.info(f"   Movies: {len(movies):,} | Ratings: {len(ratings):,} | Users: {ratings['user_id'].nunique():,}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PROCESSED FILES (used by app.py / helpers.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the already-processed CSVs. Raises FileNotFoundError if they
    don't exist yet (run process_and_save() first).
    """
    if not os.path.exists(PROCESSED_MOVIES) or not os.path.exists(PROCESSED_RATINGS):
        raise FileNotFoundError(
            "Processed Kaggle data not found. Run:\n"
            "  python backend/data/kaggle_loader.py"
        )

    movies = pd.read_csv(PROCESSED_MOVIES)
    movies["genres"] = movies["genres"].apply(
        lambda g: json.loads(g) if isinstance(g, str) and g.startswith("[") else []
    )
    movies["genres_str"] = movies["genres"].apply(lambda g: " ".join(g))
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").fillna(0).astype(int)
    movies["description"] = movies["description"].fillna("")

    ratings = pd.read_csv(PROCESSED_RATINGS)
    return movies, ratings


def kaggle_data_available() -> bool:
    """Return True if processed Kaggle data files exist."""
    return os.path.exists(PROCESSED_MOVIES) and os.path.exists(PROCESSED_RATINGS)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Kaggle movie dataset")
    parser.add_argument(
        "--max-movies", type=int, default=5000,
        help="Max number of movies to keep (default: 5000)"
    )
    parser.add_argument(
        "--dataset", choices=["movielens", "tmdb", "auto"], default="auto",
        help="Which dataset to load (default: auto-detect)"
    )
    args = parser.parse_args()

    os.makedirs(KAGGLE_DIR, exist_ok=True)
    success = process_and_save(max_movies=args.max_movies)
    if not success:
        print("\n📥 Download instructions:")
        print("  MovieLens: https://www.kaggle.com/datasets/grouplens/movielens-latest-small")
        print("  TMDB:      https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
        print("\nExtract CSV files into:  backend/data/kaggle/")
