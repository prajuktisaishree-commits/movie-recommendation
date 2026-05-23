"""
sample_data.py
--------------
Built-in sample dataset of movies and simulated user ratings.
Replace this with a CSV loader for the full MovieLens dataset.
"""

import pandas as pd
import numpy as np

MOVIES = [
    {"movie_id": 1,  "title": "The Shawshank Redemption", "genres": ["Drama"],               "year": 1994, "description": "Two imprisoned men bond over years, finding solace and redemption."},
    {"movie_id": 2,  "title": "The Godfather",             "genres": ["Crime", "Drama"],      "year": 1972, "description": "The aging patriarch of an organized crime dynasty transfers control to his son."},
    {"movie_id": 3,  "title": "The Dark Knight",           "genres": ["Action", "Crime"],     "year": 2008, "description": "Batman sets out to dismantle the remaining criminal organizations of Gotham City."},
    {"movie_id": 4,  "title": "Pulp Fiction",              "genres": ["Crime", "Drama"],      "year": 1994, "description": "Various interconnected tales of criminals in Los Angeles."},
    {"movie_id": 5,  "title": "Forrest Gump",              "genres": ["Drama", "Romance"],    "year": 1994, "description": "Forrest Gump witnesses and participates in key historical events."},
    {"movie_id": 6,  "title": "Inception",                 "genres": ["Action", "Sci-Fi"],    "year": 2010, "description": "A thief who enters dreamscapes to steal secrets is offered a chance to erase his crimes."},
    {"movie_id": 7,  "title": "The Matrix",                "genres": ["Action", "Sci-Fi"],    "year": 1999, "description": "A hacker discovers reality is a simulation and joins a rebellion."},
    {"movie_id": 8,  "title": "Goodfellas",                "genres": ["Crime", "Drama"],      "year": 1990, "description": "The story of Henry Hill and his associates in the mob."},
    {"movie_id": 9,  "title": "Interstellar",              "genres": ["Drama", "Sci-Fi"],     "year": 2014, "description": "Explorers travel through a wormhole near Saturn in search of a new home."},
    {"movie_id": 10, "title": "The Silence of the Lambs",  "genres": ["Crime", "Thriller"],   "year": 1991, "description": "An FBI trainee seeks the help of an imprisoned cannibal to catch a killer."},
    {"movie_id": 11, "title": "Schindler's List",          "genres": ["Biography", "Drama"],  "year": 1993, "description": "In Poland during WWII, a factory owner saves his Jewish workers from concentration camps."},
    {"movie_id": 12, "title": "Fight Club",                "genres": ["Drama", "Thriller"],   "year": 1999, "description": "An insomniac forms an underground fight club with a soap salesman."},
    {"movie_id": 13, "title": "The Lord of the Rings",     "genres": ["Adventure", "Fantasy"],"year": 2001, "description": "A meek hobbit embarks on a journey to destroy a dangerous ring."},
    {"movie_id": 14, "title": "Star Wars: A New Hope",     "genres": ["Action", "Sci-Fi"],    "year": 1977, "description": "A farm boy joins rebels to save a princess and destroy a space station."},
    {"movie_id": 15, "title": "The Lion King",             "genres": ["Animation", "Drama"],  "year": 1994, "description": "A young lion prince flees his kingdom only to learn the true meaning of responsibility."},
    {"movie_id": 16, "title": "Jurassic Park",             "genres": ["Adventure", "Sci-Fi"], "year": 1993, "description": "A theme park with cloned dinosaurs suffers a catastrophic breakdown."},
    {"movie_id": 17, "title": "Back to the Future",        "genres": ["Adventure", "Sci-Fi"], "year": 1985, "description": "A teen travels back in time in a time machine built by a scientist."},
    {"movie_id": 18, "title": "Titanic",                   "genres": ["Drama", "Romance"],    "year": 1997, "description": "A love story aboard the ill-fated voyage of the RMS Titanic."},
    {"movie_id": 19, "title": "Avatar",                    "genres": ["Action", "Sci-Fi"],    "year": 2009, "description": "A paraplegic marine dispatched to the moon Pandora clashes with the inhabitants."},
    {"movie_id": 20, "title": "Gladiator",                 "genres": ["Action", "Drama"],     "year": 2000, "description": "A Roman general is betrayed and seeks vengeance as a gladiator."},
    {"movie_id": 21, "title": "The Departed",              "genres": ["Crime", "Drama"],      "year": 2006, "description": "An undercover cop and a mole in the police try to identify each other."},
    {"movie_id": 22, "title": "Whiplash",                  "genres": ["Drama", "Music"],      "year": 2014, "description": "A promising young drummer enrolls in a top music conservatory."},
    {"movie_id": 23, "title": "Parasite",                  "genres": ["Drama", "Thriller"],   "year": 2019, "description": "A poor family schemes to become employed by a wealthy family."},
    {"movie_id": 24, "title": "Get Out",                   "genres": ["Horror", "Thriller"],  "year": 2017, "description": "A Black man visits his white girlfriend's family estate and discovers something sinister."},
    {"movie_id": 25, "title": "La La Land",                "genres": ["Drama", "Music"],      "year": 2016, "description": "An aspiring actress and a jazz musician fall in love in Hollywood."},
    {"movie_id": 26, "title": "Mad Max: Fury Road",        "genres": ["Action", "Adventure"], "year": 2015, "description": "In a post-apocalyptic wasteland, a woman defects from a tyrant."},
    {"movie_id": 27, "title": "The Grand Budapest Hotel",  "genres": ["Comedy", "Drama"],     "year": 2014, "description": "A concierge and his protégé become embroiled in a murder mystery."},
    {"movie_id": 28, "title": "Her",                       "genres": ["Drama", "Romance"],    "year": 2013, "description": "A lonely writer develops an unlikely relationship with an AI."},
    {"movie_id": 29, "title": "No Country for Old Men",    "genres": ["Crime", "Thriller"],   "year": 2007, "description": "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong."},
    {"movie_id": 30, "title": "The Prestige",              "genres": ["Drama", "Mystery"],    "year": 2006, "description": "Two magicians engage in a battle to create the ultimate stage illusion."},
    {"movie_id": 31, "title": "Spirited Away",             "genres": ["Animation", "Fantasy"],"year": 2001, "description": "A girl stumbles into a spirit world while moving to a new house."},
    {"movie_id": 32, "title": "WALL·E",                    "genres": ["Animation", "Sci-Fi"], "year": 2008, "description": "A small waste-collecting robot falls in love and embarks on a space journey."},
    {"movie_id": 33, "title": "Up",                        "genres": ["Animation", "Adventure"],"year": 2009,"description": "An elderly widower and a young Scout embark on a journey to South America."},
    {"movie_id": 34, "title": "Finding Nemo",              "genres": ["Animation", "Adventure"],"year": 2003,"description": "A clownfish searches for his missing son across the ocean."},
    {"movie_id": 35, "title": "The Truman Show",           "genres": ["Drama", "Sci-Fi"],     "year": 1998, "description": "An insurance salesman discovers his entire life is a reality TV show."},
    {"movie_id": 36, "title": "Memento",                   "genres": ["Mystery", "Thriller"], "year": 2000, "description": "A man with short-term memory loss investigates his wife's murder."},
    {"movie_id": 37, "title": "A Beautiful Mind",          "genres": ["Biography", "Drama"],  "year": 2001, "description": "The story of Nobel prize-winning mathematician John Nash."},
    {"movie_id": 38, "title": "Cast Away",                 "genres": ["Adventure", "Drama"],  "year": 2000, "description": "A FedEx employee must transform himself to survive after a crash on a desert island."},
    {"movie_id": 39, "title": "The Sixth Sense",           "genres": ["Drama", "Mystery"],    "year": 1999, "description": "A boy who sees dead people seeks the help of a child psychologist."},
    {"movie_id": 40, "title": "Seven",                     "genres": ["Crime", "Mystery"],    "year": 1995, "description": "Two detectives hunt a serial killer using the seven deadly sins."},
    {"movie_id": 41, "title": "Avengers: Endgame",         "genres": ["Action", "Sci-Fi"],    "year": 2019, "description": "The Avengers reassemble to undo the damage caused by Thanos."},
    {"movie_id": 42, "title": "Black Panther",             "genres": ["Action", "Adventure"], "year": 2018, "description": "T'Challa returns home to Wakanda to claim his rightful place as king."},
    {"movie_id": 43, "title": "Spider-Man: Into the Spider-Verse", "genres": ["Animation", "Action"], "year": 2018, "description": "Teen Miles Morales becomes Spider-Man and crosses paths with other Spider-heroes."},
    {"movie_id": 44, "title": "Knives Out",                "genres": ["Comedy", "Mystery"],   "year": 2019, "description": "A detective investigates the death of a crime novelist patriarch."},
    {"movie_id": 45, "title": "1917",                      "genres": ["Drama", "War"],        "year": 2019, "description": "Two soldiers are given a mission to deliver a message during WWI."},
    {"movie_id": 46, "title": "Joker",                     "genres": ["Crime", "Drama"],      "year": 2019, "description": "A failed comedian's descent into madness spawns a revolution in Gotham City."},
    {"movie_id": 47, "title": "The Irishman",              "genres": ["Crime", "Drama"],      "year": 2019, "description": "An aging hitman recalls his time with the mob and his involvement in the disappearance of Jimmy Hoffa."},
    {"movie_id": 48, "title": "Once Upon a Time in Hollywood", "genres": ["Comedy", "Drama"], "year": 2019, "description": "A faded TV actor and his stunt double navigate the changing film industry in 1969."},
    {"movie_id": 49, "title": "Dune",                      "genres": ["Adventure", "Sci-Fi"], "year": 2021, "description": "A noble family becomes embroiled in a war for the most valuable substance in the universe."},
    {"movie_id": 50, "title": "Everything Everywhere All at Once", "genres": ["Action", "Sci-Fi"], "year": 2022, "description": "An aging Chinese immigrant is swept into an adventure across the multiverse."},
]

# Generate simulated user ratings (user_id, movie_id, rating)
np.random.seed(42)

def _generate_ratings():
    """Generate realistic synthetic ratings with user taste profiles."""
    records = []
    n_users = 30

    # Each user has genre preferences
    genre_list = ["Action", "Drama", "Crime", "Sci-Fi", "Animation", "Thriller", "Comedy", "Romance", "Adventure", "Fantasy"]
    user_preferences = {
        uid: np.random.choice(genre_list, size=np.random.randint(2, 5), replace=False).tolist()
        for uid in range(1, n_users + 1)
    }

    for uid in range(1, n_users + 1):
        preferred = user_preferences[uid]
        # Rate ~60% of movies
        movies_to_rate = np.random.choice(len(MOVIES), size=np.random.randint(15, 35), replace=False)
        for idx in movies_to_rate:
            movie = MOVIES[idx]
            overlap = len(set(movie["genres"]) & set(preferred))
            base = 2.5 + overlap * 0.8
            rating = round(min(5.0, max(1.0, base + np.random.normal(0, 0.6))), 1)
            records.append({"user_id": uid, "movie_id": movie["movie_id"], "rating": rating})

    return records

RATINGS = _generate_ratings()


def get_movies_df():
    """Return movies as a DataFrame."""
    df = pd.DataFrame(MOVIES)
    df["genres_str"] = df["genres"].apply(lambda g: " ".join(g))
    return df


def get_ratings_df():
    """Return ratings as a DataFrame."""
    return pd.DataFrame(RATINGS)


def get_all_genres():
    """Return sorted list of unique genres."""
    genres = set()
    for m in MOVIES:
        genres.update(m["genres"])
    return sorted(genres)
