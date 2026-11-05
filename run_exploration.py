from src.data.loader import load_ratings
from src.exploration.analysis import (
    basic_info,
    dataset_statistics,
    rating_statistics,
    user_activity,
    item_popularity,
    popularity_threshold_analysis,
    duplicate_interaction_analysis,
    missing_value_analysis
)


DATA_PATH = "data/raw/ml-latest-small/ratings.csv"


def main():
    ratings = load_ratings(DATA_PATH)

    basic_info(ratings)
    dataset_statistics(ratings)
    rating_statistics(ratings)
    user_activity(ratings)
    item_popularity(ratings)
    popularity_threshold_analysis(ratings)
    duplicate_interaction_analysis(ratings)
    missing_value_analysis(ratings)


if __name__ == "__main__":
    main()