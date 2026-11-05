from pathlib import Path
import pandas as pd

from src.data.loader import load_ratings
from src.recommenders.user_based import (
    build_user_item_matrix,
    recommend_movies,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "raw" / "ml-latest-small"

ratings_path = DATA_DIR / "ratings.csv"
movies_path = DATA_DIR / "movies.csv"

# Load datasets
ratings = load_ratings(ratings_path)
movies = pd.read_csv(movies_path)
# Build user-item matrix
user_item_matrix = build_user_item_matrix(ratings)

# Generate recommendations
user_id = 1

recommendations = recommend_movies(
    user_item_matrix=user_item_matrix,
    movies=movies,
    user_id=user_id,
    n=10,
)

print(f"\nTop 10 recommendations for user {user_id}:\n")
print(recommendations.to_string(index=False))

from src.recommenders.user_based import calculate_user_similarity

target_mean = user_item_matrix.loc[user_id].mean()
user_means = user_item_matrix.mean(axis=1)

print("\nPrediction diagnostics:\n")

for movie_id in recommendations["movie_id"]:

    movie_ratings = user_item_matrix[movie_id].dropna()

    weighted_sum = 0
    similarity_sum = 0
    neighbor_count = 0

    for other_user_id, rating in movie_ratings.items():

        if other_user_id == user_id:
            continue

        similarity = calculate_user_similarity(
            user_item_matrix,
            user_id,
            other_user_id,
        )

        if similarity is None or similarity <= 0:
            continue

        weighted_sum += similarity * (
            rating - user_means.loc[other_user_id]
        )

        similarity_sum += similarity
        neighbor_count += 1

    raw_prediction = (
        target_mean + weighted_sum / similarity_sum
        if similarity_sum > 0 else None
    )

    print(
        f"Movie: {movie_id} | "
        f"Ratings: {len(movie_ratings)} | "
        f"Neighbors: {neighbor_count} | "
        f"Raw prediction: {raw_prediction}"
    )