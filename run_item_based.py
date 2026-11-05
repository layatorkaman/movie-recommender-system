import time
import pandas as pd

from src.data.loader import load_ratings
from src.recommenders.user_based import build_user_item_matrix
from src.recommenders.item_based import recommend_movies_item_based


ratings = load_ratings(
    "data/raw/ml-latest-small/ratings.csv"
)

movies = pd.read_csv(
    "data/raw/ml-latest-small/movies.csv"
)

user_item_matrix = build_user_item_matrix(ratings)

start_time = time.perf_counter()

recommendations = recommend_movies_item_based(
    user_item_matrix=user_item_matrix,
    movies=movies,
    user_id=1,
    n=10,
)

end_time = time.perf_counter()

print("\nItem-Based Recommendations:")
print(recommendations)

print(f"\nExecution time: {end_time - start_time:.4f} seconds")