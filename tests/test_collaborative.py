"""
test_collaborative.py
---------------------
Unit tests for the CollaborativeFilter model.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from backend.models.collaborative import CollaborativeFilter
from backend.data.sample_data import get_movies_df, get_ratings_df


@pytest.fixture
def trained_model():
    model = CollaborativeFilter()
    model.fit(get_ratings_df(), get_movies_df())
    return model


def test_fit_creates_matrix(trained_model):
    assert trained_model.user_item_matrix is not None
    assert trained_model.is_fitted is True


def test_user_item_matrix_shape(trained_model):
    # Should have at most 30 users and 50 movies
    rows, cols = trained_model.user_item_matrix.shape
    assert rows <= 30
    assert cols <= 50


def test_recommend_returns_list(trained_model):
    recs = trained_model.recommend(user_id=1, n=5)
    assert isinstance(recs, list)
    assert len(recs) == 5


def test_recommend_structure(trained_model):
    recs = trained_model.recommend(user_id=1, n=3)
    for r in recs:
        assert "movie_id" in r
        assert "title" in r
        assert "predicted_rating" in r
        assert "genres" in r


def test_recommend_excludes_seen(trained_model):
    recs = trained_model.recommend(user_id=1, n=10, exclude_seen=True)
    seen_ids = trained_model.user_item_matrix.loc[1].dropna().index.tolist()
    rec_ids = [r["movie_id"] for r in recs]
    for rid in rec_ids:
        assert rid not in seen_ids, f"Movie {rid} was already seen by user 1"


def test_recommend_invalid_user(trained_model):
    with pytest.raises(ValueError):
        trained_model.recommend(user_id=9999, n=5)


def test_get_similar_users(trained_model):
    similar = trained_model.get_similar_users(user_id=1, n=3)
    assert len(similar) == 3
    for s in similar:
        assert "user_id" in s
        assert "similarity" in s
        assert s["user_id"] != 1


def test_unfitted_model_raises():
    model = CollaborativeFilter()
    with pytest.raises(RuntimeError):
        model.recommend(user_id=1)
