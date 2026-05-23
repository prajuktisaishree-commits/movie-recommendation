#!/usr/bin/env python3
"""
process_data.py
===============
Standalone data processing pipeline for the Movie Recommendation System.

This script handles the full journey from raw Kaggle CSVs to clean,
model-ready data files.  Run it once before starting the Flask app.

Supported datasets
------------------
  1. MovieLens (small)  — 100 k ratings, 9 k movies
     Kaggle : https://www.kaggle.com/datasets/grouplens/movielens-latest-small
     Files  : movies.csv, ratings.csv, tags.csv  (tags optional)

  2. TMDB / "The Movies Dataset"  — 45 k movies, 270 k ratings
     Kaggle : https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
     Files  : movies_metadata.csv, ratings_small.csv
              credits.csv, keywords.csv  (both optional but improve quality)

Quick start
-----------
  # Option A — auto-download with Kaggle API
  pip install kaggle
  # Place kaggle.json in ~/.kaggle/  (get it from kaggle.com → Account → API)
  python process_data.py --download movielens   # or --download tmdb

  # Option B — manual download
  # Put CSVs in  backend/data/kaggle/
  python process_data.py

Output
------
  backend/data/kaggle/processed_movies.csv
  backend/data/kaggle/processed_ratings.csv
  backend/data/kaggle/processing_report.txt

Usage
-----
  python process_data.py [options]

  --download   {movielens|tmdb}   Auto-download via Kaggle API
  --dataset    {movielens|tmdb|auto}  Force a specific dataset (default: auto)
  --max-movies INT                Max movies to keep (default: 5000)
  --min-ratings INT               Min ratings a movie must have (default: 10)
  --no-report                     Skip saving the EDA report
  --force                         Re-process even if outputs already exist
"""

import os
import sys
import ast
import json
import argparse
import logging
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
KAGGLE_DIR    = ROOT_DIR / "backend" / "data" / "kaggle"
OUT_MOVIES    = KAGGLE_DIR / "processed_movies.csv"
OUT_RATINGS   = KAGGLE_DIR / "processed_ratings.csv"
OUT_REPORT    = KAGGLE_DIR / "processing_report.txt"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — AUTO-DOWNLOAD VIA KAGGLE API
# ══════════════════════════════════════════════════════════════════════════════

KAGGLE_DATASETS = {
    "movielens": "grouplens/movielens-latest-small",
    "tmdb":      "rounakbanik/the-movies-dataset",
}

def download_dataset(name: str):
    """Download a Kaggle dataset into KAGGLE_DIR using the kaggle CLI."""
    try:
        import kaggle  # noqa: F401  (just checking it's installed)
    except ImportError:
        log.error("kaggle package not installed.  Run:  pip install kaggle")
        sys.exit(1)

    slug = KAGGLE_DATASETS.get(name)
    if not slug:
        log.error(f"Unknown dataset '{name}'.  Choose: movielens | tmdb")
        sys.exit(1)

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {slug} → {KAGGLE_DIR}")
    os.system(
        f"kaggle datasets download -d {slug} "
        f"--unzip -p {KAGGLE_DIR} --force"
    )
    log.info("Download complete.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DETECT WHICH DATASET IS PRESENT
# ══════════════════════════════════════════════════════════════════════════════

def detect_dataset() -> str | None:
    if not KAGGLE_DIR.is_dir():
        return None
    files = {f.name for f in KAGGLE_DIR.iterdir()}
    if "movies_metadata.csv" in files and "ratings_small.csv" in files:
        return "tmdb"
    if "movies.csv" in files and "ratings.csv" in files:
        return "movielens"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2A — LOAD MOVIELENS
# ══════════════════════════════════════════════════════════════════════════════

def load_movielens(max_movies: int, min_ratings: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens CSVs and return (movies_df, ratings_df) in internal schema.

    Internal schema
    ---------------
    movies  : movie_id | title | genres (list) | genres_str | year | description
    ratings : user_id  | movie_id | rating
    """
    log.info("── MovieLens loader ──")

    movies_path  = KAGGLE_DIR / "movies.csv"
    ratings_path = KAGGLE_DIR / "ratings.csv"
    tags_path    = KAGGLE_DIR / "tags.csv"

    # ── 2A-1  Raw load ────────────────────────────────────────────────────────
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    log.info(f"  Raw — movies: {len(movies):,}  |  ratings: {len(ratings):,}")

    # ── 2A-2  Movies: title cleaning & year extraction ────────────────────────
    movies["year"] = (
        movies["title"]
        .str.extract(r"\((\d{4})\)\s*$")[0]
        .astype(float)
    )
    movies["title"] = (
        movies["title"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )

    # ── 2A-3  Genres: "Action|Adventure" → ["Action", "Adventure"] ───────────
    movies["genres"] = movies["genres"].apply(
        lambda g: [] if pd.isna(g) or g == "(no genres listed)" else g.split("|")
    )
    movies["genres_str"] = movies["genres"].apply(" ".join)

    # ── 2A-4  Tags: aggregate per movie → enriched description ───────────────
    if tags_path.exists():
        tags = pd.read_csv(tags_path)
        tags_agg = (
            tags.dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda x: " ".join(x.astype(str).unique()))
            .reset_index()
            .rename(columns={"tag": "tags_text"})
        )
        movies = movies.merge(tags_agg, on="movieId", how="left")
        movies["tags_text"] = movies["tags_text"].fillna("")
        log.info(f"  Tags enrichment applied ({len(tags_agg):,} movies had tags)")
    else:
        movies["tags_text"] = ""
        log.info("  tags.csv not found — skipping tag enrichment")

    movies["description"] = (movies["genres_str"] + " " + movies["tags_text"]).str.strip()

    # ── 2A-5  Ratings: filter to popular movies ───────────────────────────────
    rating_counts = ratings["movieId"].value_counts()
    popular_ids   = rating_counts[rating_counts >= min_ratings].head(max_movies).index
    ratings = ratings[ratings["movieId"].isin(popular_ids)].copy()
    movies  = movies[movies["movieId"].isin(popular_ids)].copy()
    log.info(f"  After filter — movies: {len(movies):,}  |  ratings: {len(ratings):,}")

    # ── 2A-6  Rename to internal schema ──────────────────────────────────────
    movies  = movies.rename(columns={"movieId": "movie_id"})
    ratings = ratings.rename(columns={"movieId": "movie_id", "userId": "user_id"})

    movies  = movies[["movie_id", "title", "genres", "genres_str", "year", "description"]].reset_index(drop=True)
    ratings = ratings[["user_id", "movie_id", "rating"]].reset_index(drop=True)
    return movies, ratings


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2B — LOAD TMDB
# ══════════════════════════════════════════════════════════════════════════════

def _safe_parse(val) -> list:
    """Parse a stringified Python list/dict (TMDB stores JSON as strings)."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return []


def load_tmdb(max_movies: int, min_ratings: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TMDB / 'The Movies Dataset' CSVs and return (movies_df, ratings_df).
    Merges credits and keywords when available for richer content features.
    """
    log.info("── TMDB loader ──")

    meta_path     = KAGGLE_DIR / "movies_metadata.csv"
    ratings_path  = KAGGLE_DIR / "ratings_small.csv"
    credits_path  = KAGGLE_DIR / "credits.csv"
    keywords_path = KAGGLE_DIR / "keywords.csv"

    # ── 2B-1  Metadata ────────────────────────────────────────────────────────
    meta = pd.read_csv(meta_path, low_memory=False)
    log.info(f"  Raw metadata rows: {len(meta):,}")

    # Drop adult films and rows with non-numeric ids
    meta = meta[meta["adult"] == "False"].copy()
    meta = meta[meta["id"].astype(str).str.isnumeric()].copy()
    meta["id"] = meta["id"].astype(int)

    # ── 2B-2  Genres ──────────────────────────────────────────────────────────
    meta["genres"] = meta["genres"].apply(
        lambda x: [g["name"] for g in _safe_parse(x) if isinstance(g, dict) and "name" in g]
    )
    meta["genres_str"] = meta["genres"].apply(" ".join)

    # ── 2B-3  Year ────────────────────────────────────────────────────────────
    meta["year"] = pd.to_datetime(meta["release_date"], errors="coerce").dt.year

    # ── 2B-4  Overview (base description) ────────────────────────────────────
    meta["description"] = meta["overview"].fillna("").str.strip()

    # ── 2B-5  Popularity-based movie selection ────────────────────────────────
    meta["popularity"] = pd.to_numeric(meta["popularity"], errors="coerce").fillna(0)
    meta = meta.nlargest(max_movies * 2, "popularity")   # keep extra, trim after ratings join
    log.info(f"  After popularity filter: {len(meta):,}")

    # ── 2B-6  Credits: cast + director ───────────────────────────────────────
    if credits_path.exists():
        credits = pd.read_csv(credits_path)
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
        credits = credits.dropna(subset=["id"])
        credits["id"] = credits["id"].astype(int)

        def _top_cast(val, n=5):
            actors = _safe_parse(val)
            names  = [a["name"] for a in actors[:n] if isinstance(a, dict) and "name" in a]
            return " ".join(names)

        def _director(val):
            crew = _safe_parse(val)
            dirs = [c["name"] for c in crew if isinstance(c, dict) and c.get("job") == "Director"]
            return dirs[0] if dirs else ""

        credits["cast_text"]     = credits["cast"].apply(_top_cast)
        credits["director_text"] = credits["crew"].apply(_director)
        meta = meta.merge(credits[["id", "cast_text", "director_text"]], on="id", how="left")
        meta["cast_text"]     = meta["cast_text"].fillna("")
        meta["director_text"] = meta["director_text"].fillna("")
        log.info("  Credits merged (cast + director)")
    else:
        meta["cast_text"]     = ""
        meta["director_text"] = ""
        log.info("  credits.csv not found — skipping cast enrichment")

    # ── 2B-7  Keywords ────────────────────────────────────────────────────────
    if keywords_path.exists():
        kw = pd.read_csv(keywords_path)
        kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
        kw = kw.dropna(subset=["id"])
        kw["id"] = kw["id"].astype(int)
        kw["kw_text"] = kw["keywords"].apply(
            lambda x: " ".join(k["name"] for k in _safe_parse(x) if isinstance(k, dict) and "name" in k)
        )
        meta = meta.merge(kw[["id", "kw_text"]], on="id", how="left")
        meta["kw_text"] = meta["kw_text"].fillna("")
        log.info("  Keywords merged")
    else:
        meta["kw_text"] = ""
        log.info("  keywords.csv not found — skipping keyword enrichment")

    # ── 2B-8  Combined description (all text features) ───────────────────────
    meta["description"] = (
        meta["description"]      + " "
        + meta["genres_str"]     + " "
        + meta["cast_text"]      + " "
        + meta["director_text"]  + " "
        + meta["kw_text"]
    ).str.strip()

    # ── 2B-9  Ratings ─────────────────────────────────────────────────────────
    ratings = pd.read_csv(ratings_path)
    ratings = ratings.rename(columns={"movieId": "movie_id", "userId": "user_id"})
    log.info(f"  Raw ratings: {len(ratings):,}")

    valid_ids = set(meta["id"])
    ratings   = ratings[ratings["movie_id"].isin(valid_ids)].copy()

    # Keep only movies with enough ratings
    counts   = ratings["movie_id"].value_counts()
    keep_ids = set(counts[counts >= min_ratings].head(max_movies).index)
    ratings  = ratings[ratings["movie_id"].isin(keep_ids)].copy()
    meta     = meta[meta["id"].isin(keep_ids)].copy()
    log.info(f"  After filter — movies: {len(meta):,}  |  ratings: {len(ratings):,}")

    # ── 2B-10  Final schema ───────────────────────────────────────────────────
    meta = meta.rename(columns={"id": "movie_id"})
    meta["title"] = meta["title"].fillna(meta.get("original_title", "Unknown")).fillna("Unknown")

    movies  = meta[["movie_id", "title", "genres", "genres_str", "year", "description"]].reset_index(drop=True)
    ratings = ratings[["user_id", "movie_id", "rating"]].reset_index(drop=True)
    return movies, ratings


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SHARED CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean(movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply shared post-load cleaning to both DataFrames.

    Operations
    ----------
    Movies  : deduplicate, fill missing years, cap description length
    Ratings : clip ratings to [0.5, 5.0], remove duplicates, drop NaN
    """
    log.info("── Cleaning ──")

    # ── Movies ────────────────────────────────────────────────────────────────
    before = len(movies)
    movies = movies.drop_duplicates(subset=["movie_id"])
    movies = movies.drop_duplicates(subset=["title"])
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").fillna(0).astype(int)
    movies["title"] = movies["title"].str.strip()
    movies["description"] = movies["description"].fillna("").str.slice(0, 2000)  # cap at 2 k chars
    log.info(f"  Movies: {before:,} → {len(movies):,} (removed {before - len(movies):,} dupes)")

    # ── Ratings ───────────────────────────────────────────────────────────────
    before = len(ratings)
    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])
    ratings["rating"] = ratings["rating"].astype(float).clip(0.5, 5.0)
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    # Keep only ratings for movies that survived cleaning
    ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]
    log.info(f"  Ratings: {before:,} → {len(ratings):,} (removed {before - len(ratings):,})")

    return movies.reset_index(drop=True), ratings.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def feature_engineering(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add derived features used by the recommendation models.

    Movies
    ------
    avg_rating     : mean rating across all users
    rating_count   : number of ratings received
    bayesian_score : shrinkage-adjusted score (better than raw mean for sparse items)

    Ratings
    -------
    rating_norm    : per-user mean-centred rating  (helps collaborative filter)
    """
    log.info("── Feature engineering ──")

    # ── Per-movie stats ───────────────────────────────────────────────────────
    stats = (
        ratings.groupby("movie_id")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    stats["avg_rating"]   = stats["avg_rating"].round(3)

    # Bayesian average:  (C * m + Σr) / (C + n)
    # where C = min_votes threshold, m = global mean
    C = stats["rating_count"].quantile(0.25)   # 25th-percentile vote count
    m = ratings["rating"].mean()
    stats["bayesian_score"] = (
        (C * m + stats["avg_rating"] * stats["rating_count"])
        / (C + stats["rating_count"])
    ).round(3)

    movies = movies.merge(stats, on="movie_id", how="left")
    movies["avg_rating"]    = movies["avg_rating"].fillna(m).round(3)
    movies["rating_count"]  = movies["rating_count"].fillna(0).astype(int)
    movies["bayesian_score"]= movies["bayesian_score"].fillna(m).round(3)

    log.info(f"  Global mean rating   : {m:.3f}")
    log.info(f"  Bayesian C threshold : {C:.0f} votes")

    # ── Per-user mean-centred ratings ─────────────────────────────────────────
    user_means = ratings.groupby("user_id")["rating"].mean().rename("user_mean")
    ratings    = ratings.join(user_means, on="user_id")
    ratings["rating_norm"] = (ratings["rating"] - ratings["user_mean"]).round(4)
    ratings = ratings.drop(columns=["user_mean"])

    log.info("  Per-user normalised ratings added")
    return movies, ratings


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(movies: pd.DataFrame, ratings: pd.DataFrame) -> bool:
    """
    Run sanity checks.  Logs warnings for soft failures, returns False for
    hard failures that would break the models.
    """
    log.info("── Validation ──")
    ok = True

    # Hard checks
    for col in ["movie_id", "title", "genres", "description"]:
        if col not in movies.columns:
            log.error(f"  movies missing required column: {col}")
            ok = False

    for col in ["user_id", "movie_id", "rating"]:
        if col not in ratings.columns:
            log.error(f"  ratings missing required column: {col}")
            ok = False

    if not ok:
        return False

    # Soft checks
    null_titles = movies["title"].isna().sum()
    if null_titles:
        log.warning(f"  {null_titles} movies have null titles")

    null_genres = movies["genres"].apply(lambda g: len(g) == 0).sum()
    if null_genres > len(movies) * 0.5:
        log.warning(f"  {null_genres}/{len(movies)} movies have no genres")

    out_of_range = ((ratings["rating"] < 0.5) | (ratings["rating"] > 5.0)).sum()
    if out_of_range:
        log.warning(f"  {out_of_range} ratings outside [0.5, 5.0]")

    orphan_ratings = (~ratings["movie_id"].isin(movies["movie_id"])).sum()
    if orphan_ratings:
        log.warning(f"  {orphan_ratings} ratings reference unknown movie_ids")

    log.info(f"  ✓ movies: {len(movies):,}  |  ratings: {len(ratings):,}  |  users: {ratings['user_id'].nunique():,}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EDA REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(movies: pd.DataFrame, ratings: pd.DataFrame, dataset: str) -> str:
    """Build a plain-text EDA report and return it as a string."""
    from collections import Counter

    lines = []
    sep   = "═" * 60

    def h(title): lines.extend([sep, f"  {title}", sep])
    def kv(k, v): lines.append(f"  {k:<30} {v}")
    def br():     lines.append("")

    h("PROCESSING REPORT")
    kv("Generated",       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kv("Dataset",         dataset.upper())
    br()

    h("DATASET OVERVIEW")
    kv("Total movies",    f"{len(movies):,}")
    kv("Total ratings",   f"{len(ratings):,}")
    kv("Total users",     f"{ratings['user_id'].nunique():,}")
    kv("Avg ratings/user",f"{len(ratings)/ratings['user_id'].nunique():.1f}")
    kv("Avg ratings/movie",f"{len(ratings)/len(movies):.1f}")
    br()

    h("RATING DISTRIBUTION")
    for star in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        cnt = (ratings["rating"] == star).sum()
        bar = "█" * int(cnt / len(ratings) * 40)
        kv(f"  {star:.1f} ★", f"{cnt:>7,}  {bar}")
    kv("Mean rating",     f"{ratings['rating'].mean():.3f}")
    kv("Std deviation",   f"{ratings['rating'].std():.3f}")
    br()

    h("TOP 10 GENRES")
    genre_counts = Counter(g for gs in movies["genres"] for g in gs)
    for genre, cnt in genre_counts.most_common(10):
        bar = "█" * int(cnt / max(genre_counts.values()) * 30)
        kv(f"  {genre}", f"{cnt:>5,}  {bar}")
    br()

    h("TOP 20 MOST-RATED MOVIES")
    top = (
        ratings.groupby("movie_id")["rating"]
        .agg(count="count", mean="mean")
        .join(movies.set_index("movie_id")[["title"]])
        .sort_values("count", ascending=False)
        .head(20)
    )
    for _, row in top.iterrows():
        kv(f"  {str(row['title'])[:35]:<35}", f"{int(row['count']):>6,} ratings  ★{row['mean']:.2f}")
    br()

    h("YEAR DISTRIBUTION (decade bins)")
    years = movies[movies["year"] > 0]["year"]
    for decade in range(1920, 2030, 10):
        cnt = ((years >= decade) & (years < decade + 10)).sum()
        if cnt:
            bar = "█" * int(cnt / len(years) * 35)
            kv(f"  {decade}s", f"{cnt:>5,}  {bar}")
    br()

    h("DATA QUALITY")
    kv("Movies with genres",      f"{movies['genres'].apply(bool).sum():,}")
    kv("Movies without genres",   f"{ (~movies['genres'].apply(bool)).sum():,}")
    kv("Movies with description", f"{(movies['description'].str.len() > 10).sum():,}")
    kv("Movies with year",        f"{(movies['year'] > 0).sum():,}")
    br()

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save(movies: pd.DataFrame, ratings: pd.DataFrame, report: str, save_report: bool):
    """Serialise and write output files."""
    log.info("── Saving ──")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    # Genres: list → JSON string (CSV can't store lists natively)
    movies_out = movies.copy()
    movies_out["genres"] = movies_out["genres"].apply(
        lambda g: json.dumps(g) if isinstance(g, list) else g
    )
    movies_out.to_csv(OUT_MOVIES, index=False)
    ratings.to_csv(OUT_RATINGS, index=False)
    log.info(f"  Saved: {OUT_MOVIES}")
    log.info(f"  Saved: {OUT_RATINGS}")

    if save_report:
        OUT_REPORT.write_text(report, encoding="utf-8")
        log.info(f"  Saved: {OUT_REPORT}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    dataset:    str  = "auto",
    max_movies: int  = 5000,
    min_ratings: int = 10,
    save_report: bool = True,
    force:       bool = False,
):
    log.info("=" * 60)
    log.info("  Movie Recommender — Data Processing Pipeline")
    log.info("=" * 60)

    # Early exit if outputs already exist
    if not force and OUT_MOVIES.exists() and OUT_RATINGS.exists():
        log.info("Processed files already exist.  Use --force to reprocess.")
        return True

    # Detect dataset
    if dataset == "auto":
        dataset = detect_dataset()
        if dataset is None:
            log.error(
                textwrap.dedent(f"""
                No Kaggle CSV files found in: {KAGGLE_DIR}

                Download one of the supported datasets:
                  MovieLens : https://www.kaggle.com/datasets/grouplens/movielens-latest-small
                  TMDB      : https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

                Then place the CSV files in:  {KAGGLE_DIR}
                Or use --download movielens / --download tmdb  (requires kaggle package + API key)
                """).strip()
            )
            return False
    log.info(f"Dataset: {dataset.upper()}")

    # Load
    if dataset == "movielens":
        movies, ratings = load_movielens(max_movies, min_ratings)
    else:
        movies, ratings = load_tmdb(max_movies, min_ratings)

    # Clean
    movies, ratings = clean(movies, ratings)

    # Feature engineering
    movies, ratings = feature_engineering(movies, ratings)

    # Validate
    if not validate(movies, ratings):
        log.error("Validation failed — aborting.")
        return False

    # Report
    report = generate_report(movies, ratings, dataset)
    print("\n" + report + "\n")

    # Save
    save(movies, ratings, report, save_report)

    log.info("=" * 60)
    log.info("  Pipeline complete ✅")
    log.info(f"  Movies  : {len(movies):,}")
    log.info(f"  Ratings : {len(ratings):,}")
    log.info(f"  Users   : {ratings['user_id'].nunique():,}")
    log.info("=" * 60)
    log.info("You can now start the app:  cd backend && python app.py")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Kaggle movie data for the recommendation system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          # Auto-detect dataset already in backend/data/kaggle/
          python process_data.py

          # Auto-download MovieLens via Kaggle API then process
          python process_data.py --download movielens

          # TMDB dataset, keep 10 000 movies, min 20 ratings
          python process_data.py --download tmdb --max-movies 10000 --min-ratings 20

          # Force re-process even if output files already exist
          python process_data.py --force
        """),
    )

    parser.add_argument(
        "--download", choices=["movielens", "tmdb"], default=None,
        metavar="DATASET",
        help="Auto-download dataset via Kaggle API before processing",
    )
    parser.add_argument(
        "--dataset", choices=["movielens", "tmdb", "auto"], default="auto",
        help="Which dataset to process (default: auto-detect)",
    )
    parser.add_argument(
        "--max-movies", type=int, default=5000,
        help="Maximum number of movies to keep (default: 5000)",
    )
    parser.add_argument(
        "--min-ratings", type=int, default=10,
        help="Minimum ratings a movie must have (default: 10)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip saving the EDA report to disk",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if output files already exist",
    )

    args = parser.parse_args()

    if args.download:
        download_dataset(args.download)
        if args.dataset == "auto":
            args.dataset = args.download

    success = run(
        dataset     = args.dataset,
        max_movies  = args.max_movies,
        min_ratings = args.min_ratings,
        save_report = not args.no_report,
        force       = args.force,
    )
    sys.exit(0 if success else 1)
