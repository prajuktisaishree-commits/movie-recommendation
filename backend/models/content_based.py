"""
content_based.py
----------------
Content-Based Filtering using TF-IDF on movie metadata
(genres, description) and cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """
    Content-Based Filtering: recommends movies similar to a seed movie
    based on genres and description text.
    """

    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.is_fitted = False

    def fit(self, movies_df: pd.DataFrame):
        """
        Train the model by computing TF-IDF vectors for each movie.

        Parameters
        ----------
        movies_df : pd.DataFrame  — must have columns: movie_id, title, genres, description
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)

        # Combine genres and description into one feature string
        self.movies_df["soup"] = (
            self.movies_df["genres_str"].fillna("") + " "
            + self.movies_df["description"].fillna("")
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["soup"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Map title → index
        self.indices = pd.Series(
            self.movies_df.index, index=self.movies_df["title"]
        ).drop_duplicates()

        self.is_fitted = True
        return self

    def recommend_by_movie(self, title: str, n: int = 10):
        """
        Recommend movies similar to the given title.

        Parameters
        ----------
        title : str — exact movie title (case-insensitive match attempted)
        n     : int — number of recommendations

        Returns
        -------
        list of dicts [{movie_id, title, similarity_score, genres}, ...]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Case-insensitive title lookup
        title_lower = title.lower()
        matches = self.movies_df[self.movies_df["title"].str.lower() == title_lower]
        if matches.empty:
            # Partial match fallback
            matches = self.movies_df[self.movies_df["title"].str.lower().str.contains(title_lower)]
        if matches.empty:
            raise ValueError(f"Movie '{title}' not found.")

        idx = matches.index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        results = []
        for i, score in sim_scores:
            row = self.movies_df.iloc[i]
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "similarity_score": round(float(score), 4),
                "genres": row["genres"],
                "year": int(row["year"]),
            })
        return results

    def recommend_by_genres(self, genres: list, n: int = 10):
        """
        Recommend movies matching given genres.

        Parameters
        ----------
        genres : list of str — e.g. ["Action", "Sci-Fi"]
        n      : int         — number of results

        Returns
        -------
        list of dicts sorted by genre overlap count
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        genre_set = set(g.lower() for g in genres)

        def overlap(row_genres):
            return len({g.lower() for g in row_genres} & genre_set)

        df = self.movies_df.copy()
        df["overlap"] = df["genres"].apply(overlap)
        df = df[df["overlap"] > 0].sort_values("overlap", ascending=False)

        results = []
        for _, row in df.head(n).iterrows():
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "year": int(row["year"]),
                "overlap": int(row["overlap"]),
            })
        return results

    def get_movie_vector(self, title: str):
        """Return the top TF-IDF keywords for a movie (for explanation)."""
        title_lower = title.lower()
        matches = self.movies_df[self.movies_df["title"].str.lower() == title_lower]
        if matches.empty:
            return []
        idx = matches.index[0]
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_row = self.tfidf_matrix[idx].toarray().flatten()
        top_indices = tfidf_row.argsort()[::-1][:10]
        return [(feature_names[i], round(float(tfidf_row[i]), 4)) for i in top_indices if tfidf_row[i] > 0]
