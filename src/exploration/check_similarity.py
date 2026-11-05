
from src.data.loader import load_ratings
from src.recommenders.user_based import (
    build_user_item_matrix,
    calculate_user_similarity,
)


DATA_PATH = "data/raw/ml-latest-small/ratings.csv"


def main():
    ratings = load_ratings(DATA_PATH)
    matrix = build_user_item_matrix(ratings)

    user_id = 1
    other_user_id = 2

    common_ratings = matrix.loc[
        [user_id, other_user_id]
    ].dropna(axis=1)

    print("User 1:", user_id)
    print("User 2:", other_user_id)
    print("Number of common movies:", common_ratings.shape[1])

    similarity = calculate_user_similarity(
        matrix,
        user_id=user_id,
        other_user_id=other_user_id,
        min_common_ratings=3,
    )

    print("Pearson similarity:", similarity)


if __name__ == "__main__":
    main()
