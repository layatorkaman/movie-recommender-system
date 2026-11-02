import pandas as pd


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user-item matrix from ratings data.

    Rows represent users.
    Columns represent movies.
    Values represent ratings.
    """
    user_item_matrix = ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating",
    )

    return user_item_matrix


def calculate_user_similarity(
        user_item_matrix: pd.DataFrame,
        user_id: int,
        other_user_id: int,
        min_common_ratings: int = 3,
) -> float | None:
    """
    Calculate Pearson similarity between two users.

    Return None when there are not enough common ratings
    or when Pearson correlation cannot be calculated.
    """
    common_ratings = user_item_matrix.loc[
        [user_id, other_user_id]
    ].dropna(axis=1)

    if common_ratings.shape[1] < min_common_ratings:
        return None

    similarity = common_ratings.loc[user_id].corr(
        common_ratings.loc[other_user_id],
        method="pearson",
    )

    if pd.isna(similarity):
        return None

    return float(similarity)



def predict_rating(
    user_item_matrix: pd.DataFrame,
    user_id: int,
    movie_id: int,
    min_common_ratings: int = 3,
) -> float | None:
    """
    Predict a user's rating for a movie using
    mean-centered similarity-weighted ratings.
    """

    if user_id not in user_item_matrix.index:
        return None

    if movie_id not in user_item_matrix.columns:
        return None

    # Calculate the target user's mean rating.
    target_user_mean = user_item_matrix.loc[user_id].mean()

    # Find users who have rated the target movie.
    users_who_rated_movie = user_item_matrix.index[
        user_item_matrix[movie_id].notna()
    ]

    weighted_sum = 0.0
    similarity_sum = 0.0

    for other_user_id in users_who_rated_movie:

        if other_user_id == user_id:
            continue

        similarity = calculate_user_similarity(
            user_item_matrix,
            user_id,
            other_user_id,
            min_common_ratings=min_common_ratings,
        )

        if similarity is None or similarity <= 0:
            continue

        rating = user_item_matrix.loc[other_user_id, movie_id]

        # Calculate the neighbor's mean rating.
        other_user_mean = user_item_matrix.loc[other_user_id].mean()

        # Add the weighted deviation from the neighbor's mean.
        weighted_sum += similarity * (rating - other_user_mean)
        similarity_sum += similarity

    if similarity_sum == 0:
        return None

    predicted_rating = (
        target_user_mean + weighted_sum / similarity_sum
    )

    # Keep the prediction within the valid rating range.
    return float(max(0.5, min(5.0, predicted_rating)))


def recommend_movies(
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 10,
    min_common_ratings: int = 3,
) -> pd.DataFrame:
    """
    Recommend the top-N unseen movies for a user.
    """

    result_columns = ["movie_id", "title", "predicted_rating"]

    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=result_columns)

    required_columns = {"movieId", "title"}

    if not required_columns.issubset(movies.columns):
        raise ValueError("Movies DataFrame must contain movieId and title.")

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    # Find movies the user has not rated yet.
    unseen_movie_ids = user_item_matrix.columns[
        user_item_matrix.loc[user_id].isna()
    ]

    recommendations = []

    for movie_id in unseen_movie_ids:

        predicted_rating = predict_rating(
            user_item_matrix,
            user_id,
            movie_id,
            min_common_ratings=min_common_ratings,
        )

        if predicted_rating is None:
            continue

        movie_info = movies.loc[movies["movieId"] == movie_id]

        if movie_info.empty:
            continue

        title = movie_info.iloc[0]["title"]

        recommendations.append({
            "movie_id": movie_id,
            "title": title,
            "predicted_rating": predicted_rating,
        })

    recommendations_df = pd.DataFrame(
        recommendations,
        columns=result_columns,
    )

    return (
        recommendations_df
        .sort_values("predicted_rating", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def test_recommend_movies():
    from src.recommenders.user_based import recommend_movies

    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "movieId": [10, 20, 30, 10, 20, 40, 10, 20, 50],
        "rating":  [5, 4, 1, 5, 4, 5, 4, 3, 5],
    })

    movies = pd.DataFrame({
        "movieId": [10, 20, 30, 40, 50],
        "title": [
            "Movie A",
            "Movie B",
            "Movie C",
            "Movie D",
            "Movie E",
        ],
    })

    matrix = build_user_item_matrix(ratings)

    recommendations = recommend_movies(
        matrix,
        movies,
        user_id=1,
        n=2,
        min_common_ratings=2,
    )

    assert len(recommendations) == 2
    assert list(recommendations.columns) == [
        "movie_id",
        "title",
        "predicted_rating",
    ]
    assert 10 not in recommendations["movie_id"].values
    assert 20 not in recommendations["movie_id"].values
    assert recommendations["predicted_rating"].is_monotonic_decreasing
