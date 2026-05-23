"""
test_content_based.py
---------------------
Unit tests for the ContentBasedFilter model.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from backend.models.content_based import ContentBasedFilter
from backend.data.sample_data import get_movies_df


@pytest.fixture
def trained_model():
    model = ContentBasedFilter()
    model.fit(get_movies_df())
    return model


def test_fit_creates_matrix(trained_model):
    assert trained_model.tfidf_matrix is not None
    assert trained_model.cosine_sim is not None
    assert trained_model.is_fitted is True


def test_cosine_sim_shape(trained_model):
    n = len(get_movies_df())
    assert trained_model.cosine_sim.shape == (n, n)


def test_recommend_by_movie_returns_list(trained_model):
    recs = trained_model.recommend_by_movie("Inception", n=5)
    assert isinstance(recs, list)
    assert len(recs) == 5


def test_recommend_excludes_seed(trained_model):
    recs = trained_model.recommend_by_movie("Inception", n=10)
    titles = [r["title"] for r in recs]
    assert "Inception" not in titles


def test_recommend_structure(trained_model):
    recs = trained_model.recommend_by_movie("The Dark Knight", n=3)
    for r in recs:
        assert "movie_id" in r
        assert "title" in r
        assert "similarity_score" in r
        assert 0.0 <= r["similarity_score"] <= 1.0


def test_recommend_by_genre(trained_model):
    recs = trained_model.recommend_by_genres(["Sci-Fi"], n=5)
    assert len(recs) >= 1
    for r in recs:
        assert "Sci-Fi" in r["genres"]


def test_recommend_multiple_genres(trained_model):
    recs = trained_model.recommend_by_genres(["Action", "Sci-Fi"], n=8)
    assert len(recs) >= 1


def test_invalid_movie_raises(trained_model):
    with pytest.raises(ValueError):
        trained_model.recommend_by_movie("NONEXISTENT MOVIE XYZ")


def test_unfitted_model_raises():
    model = ContentBasedFilter()
    with pytest.raises(RuntimeError):
        model.recommend_by_movie("Inception")


def test_case_insensitive_lookup(trained_model):
    recs = trained_model.recommend_by_movie("inception", n=3)
    assert len(recs) == 3


def test_get_movie_vector(trained_model):
    keywords = trained_model.get_movie_vector("Inception")
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)
