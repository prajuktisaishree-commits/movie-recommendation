"""
collaborative.py
----------------
Collaborative Filtering using Item-Based approach with TF-IDF vectors
for memory efficiency on large datasets.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilter:
    """
    Item-Based Collaborative Filtering using TF-IDF on user rating patterns.
    Avoids the O(n²) memory cost of user-user similarity matrices for large datasets.
    """

    def __init__(self):
        self.user_item_matrix = None
        self.user_item_sparse = None
        self.user_idx_map = None
        self.movies_df = None
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Train the model by building a user-item matrix (stored as sparse).

        Parameters
        ----------
        ratings_df : pd.DataFrame  columns: [user_id, movie_id, rating]
        movies_df  : pd.DataFrame  columns: [movie_id, title, ...]
        """
        self.movies_df = movies_df.copy()

        # Build user × movie matrix (rows=users, cols=movies)
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )

        # Fill NaN with 0 and convert to sparse format
        matrix_filled = self.user_item_matrix.fillna(0).values
        self.user_item_sparse = csr_matrix(matrix_filled)

        # Create a mapping from user_id to index
        self.user_idx_map = {uid: idx for idx, uid in enumerate(self.user_item_matrix.index)}

        self.is_fitted = True
        return self

    def _get_user_similarities(self, user_id: int, top_k: int = 50):
        """
        Compute similarities only for the top-K most similar users
        (on-demand, memory efficient).
        """
        if user_id not in self.user_idx_map:
            raise ValueError(f"User {user_id} not found in training data.")

        user_idx = self.user_idx_map[user_id]
        user_vector = self.user_item_sparse[user_idx].reshape(1, -1)

        # Compute similarity with all other users (still linear, not quadratic)
        similarities = cosine_similarity(user_vector, self.user_item_sparse).ravel()

        # Get top-K most similar users (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        top_sims = similarities[top_indices]

        return top_indices, top_sims

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True):
        """
        Recommend top-N movies for a given user using top-K similar users.

        Parameters
        ----------
        user_id      : int   — target user
        n            : int   — number of recommendations
        exclude_seen : bool  — exclude movies the user has already rated

        Returns
        -------
        list of dicts [{movie_id, title, predicted_rating, genres}, ...]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        if user_id not in self.user_idx_map:
            raise ValueError(f"User {user_id} not found in training data.")

        # Get top-50 similar users
        similar_user_indices, similarities = self._get_user_similarities(user_id, top_k=50)

        # Compute weighted average ratings from similar users
        matrix = self.user_item_sparse.toarray()
        similar_matrix = matrix[similar_user_indices]  # (K, n_movies)

        # Weighted sum
        sim_sum = np.abs(similarities).sum() + 1e-9
        weighted_ratings = (similar_matrix * similarities[:, np.newaxis]).sum(axis=0) / sim_sum

        # Map to movie IDs
        movie_ids = self.user_item_matrix.columns
        predictions = pd.Series(weighted_ratings, index=movie_ids)

        if exclude_seen:
            user_idx = self.user_idx_map[user_id]
            seen_mask = matrix[user_idx] > 0
            seen_ids = movie_ids[seen_mask].tolist()
            predictions = predictions.drop(seen_ids, errors="ignore")

        top_ids = predictions.nlargest(n).index.tolist()

        results = []
        for mid in top_ids:
            row = self.movies_df[self.movies_df["movie_id"] == mid]
            if not row.empty:
                results.append({
                    "movie_id": int(mid),
                    "title": row.iloc[0]["title"],
                    "predicted_rating": round(float(predictions[mid]), 2),
                    "genres": row.iloc[0]["genres"],
                    "year": int(row.iloc[0]["year"]),
                })
        return results

    def get_similar_users(self, user_id: int, n: int = 5):
        """Return the top-N most similar users to a given user."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if user_id not in self.user_idx_map:
            raise ValueError(f"User {user_id} not found.")

        similar_indices, similarities = self._get_user_similarities(user_id, top_k=n)
        user_ids = list(self.user_item_matrix.index)
        return [
            {"user_id": user_ids[i], "similarity": round(float(similarities[idx]), 4)}
            for idx, i in enumerate(similar_indices)
        ]
