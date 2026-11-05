import pandas as pd


def basic_info(ratings: pd.DataFrame) -> None:
    print("Dataset shape:")
    print(ratings.shape)

    print("\nColumns:")
    print(ratings.columns)

    print("\nFirst rows:")
    print(ratings.head())


def dataset_statistics(ratings: pd.DataFrame) -> None:
    print("Number of users:")
    print(ratings["userId"].nunique())

    print("\nNumber of movies:")
    print(ratings["movieId"].nunique())

    print("\nNumber of ratings:")
    print(len(ratings))


def rating_statistics(ratings: pd.DataFrame) -> None:
    print("Average rating:")
    print(ratings["rating"].mean())

    print("\nMinimum rating:")
    print(ratings["rating"].min())

    print("\nMaximum rating:")
    print(ratings["rating"].max())

    print("\nRating distribution:")
    print(ratings["rating"].value_counts().sort_index())


def user_activity(ratings: pd.DataFrame) -> None:
    user_rating_counts = ratings.groupby("userId").size()

    print("\nUser activity:")
    print(user_rating_counts.describe())

def item_popularity(ratings: pd.DataFrame) -> None:
    item_popularity_count = ratings.groupby("movieId").size()
    print("\nItem Popularity:")
    print(item_popularity_count.describe())

def popularity_threshold_analysis(ratings: pd.DataFrame) -> None:
    item_rating_counts = ratings.groupby("movieId").size()

    for threshold in [5, 10, 20, 50, 100]:
        count = (item_rating_counts >= threshold).sum()
        print(f"At least {threshold} ratings: {count} movies")



def duplicate_interaction_analysis(ratings: pd.DataFrame) -> None:
    duplicate_count = ratings.duplicated(
        subset=["userId", "movieId"]
    ).sum()

    print("\nDuplicate interactions:")
    print(duplicate_count)

def missing_value_analysis(ratings: pd.DataFrame) -> None:
    missing_values = ratings.isna().sum()

    print("\nMissing values:")
    print(missing_values)
