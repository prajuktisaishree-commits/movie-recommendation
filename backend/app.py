"""
app.py
------
Flask REST API for the Movie Recommendation System.
Automatically uses Kaggle data if processed files exist,
otherwise falls back to the built-in sample dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # type: ignore[import]

from backend.utils.helpers import (
    get_trained_models, get_data_source, serialize_movie, get_movies_and_ratings
)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
)
CORS(app)


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ──────────────────────────────────────────────
# API — Info
# ──────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    source = get_data_source()
    movies_df, ratings_df = get_movies_and_ratings()
    return jsonify({
        "status": "ok",
        "data_source": source,
        "movies": len(movies_df),
        "ratings": len(ratings_df),
        "users": ratings_df["user_id"].nunique(),
    })

@app.route("/api/datasource", methods=["GET"])
def datasource():
    """Return which data source is active and basic stats."""
    source = get_data_source()
    movies_df, ratings_df = get_movies_and_ratings()
    return jsonify({
        "source": source,
        "label": "Kaggle Dataset" if source == "kaggle" else "Built-in Sample Data",
        "movies": len(movies_df),
        "ratings": len(ratings_df),
        "users": int(ratings_df["user_id"].nunique()),
    })


# ──────────────────────────────────────────────
# API — Movies
# ──────────────────────────────────────────────

@app.route("/api/movies", methods=["GET"])
def get_movies():
    """Return movies, optionally filtered by genre. Supports pagination."""
    genre  = request.args.get("genre", "").strip()
    page   = int(request.args.get("page", 1))
    limit  = min(int(request.args.get("limit", 200)), 500)

    movies_df, _ = get_movies_and_ratings()
    df = movies_df.copy()

    if genre:
        df = df[df["genres"].apply(lambda g: genre in g)]

    total = len(df)
    df = df.iloc[(page - 1) * limit : page * limit]

    movies_list = []
    for _, row in df.iterrows():
        movies_list.append({
            "movie_id": int(row["movie_id"]),
            "title":    str(row["title"]),
            "genres":   list(row["genres"]),
            "year":     int(row["year"]) if row["year"] else 0,
        })

    return jsonify({"movies": movies_list, "total": total, "page": page})


@app.route("/api/genres", methods=["GET"])
def get_genres():
    """Return all available genres from the active dataset."""
    movies_df, _ = get_movies_and_ratings()
    genres = sorted(set(g for genres in movies_df["genres"] for g in genres if g))
    return jsonify({"genres": genres})


# ──────────────────────────────────────────────
# API — Collaborative Filtering
# ──────────────────────────────────────────────

@app.route("/api/recommend/collaborative", methods=["GET"])
def recommend_collaborative():
    """GET /api/recommend/collaborative?user_id=1&n=10"""
    try:
        user_id = int(request.args.get("user_id", 1))
        n = min(int(request.args.get("n", 10)), 20)
    except (ValueError, TypeError):
        return jsonify({"error": "user_id and n must be integers."}), 400

    collaborative, _ = get_trained_models()
    try:
        recs = collaborative.recommend(user_id=user_id, n=n)
        return jsonify({
            "user_id": user_id,
            "method": "collaborative_filtering",
            "data_source": get_data_source(),
            "recommendations": [serialize_movie(r) for r in recs],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500


@app.route("/api/similar-users", methods=["GET"])
def similar_users():
    """GET /api/similar-users?user_id=1&n=5"""
    try:
        user_id = int(request.args.get("user_id", 1))
        n = int(request.args.get("n", 5))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid parameters."}), 400

    collaborative, _ = get_trained_models()
    try:
        similar = collaborative.get_similar_users(user_id=user_id, n=n)
        return jsonify({"user_id": user_id, "similar_users": similar})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ──────────────────────────────────────────────
# API — Content-Based Filtering
# ──────────────────────────────────────────────

@app.route("/api/recommend/content", methods=["GET"])
def recommend_content():
    """GET /api/recommend/content?title=Inception&n=10"""
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "title parameter is required."}), 400
    try:
        n = min(int(request.args.get("n", 10)), 20)
    except (ValueError, TypeError):
        n = 10

    _, content = get_trained_models()
    try:
        recs = content.recommend_by_movie(title=title, n=n)
        return jsonify({
            "seed_movie": title,
            "method": "content_based_filtering",
            "data_source": get_data_source(),
            "recommendations": [serialize_movie(r) for r in recs],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500


@app.route("/api/recommend/genre", methods=["GET"])
def recommend_by_genre():
    """GET /api/recommend/genre?genres=Action,Sci-Fi&n=10"""
    genres_raw = request.args.get("genres", "").strip()
    if not genres_raw:
        return jsonify({"error": "genres parameter is required."}), 400

    genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
    try:
        n = min(int(request.args.get("n", 10)), 20)
    except (ValueError, TypeError):
        n = 10

    _, content = get_trained_models()
    try:
        recs = content.recommend_by_genres(genres=genres, n=n)
        return jsonify({
            "genres": genres,
            "method": "genre_filtering",
            "data_source": get_data_source(),
            "recommendations": [serialize_movie(r) for r in recs],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🎬 Movie Recommendation System starting...")
    print("   Visit http://localhost:5000")
    app.run(debug=True, port=5000)
