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

    common_ratings = user_item_matrix.loc[
        [user_id, other_user_id]
    ].dropna(axis=1)

    if common_ratings.shape[1] < min_common_ratings:
        return None

    user_ratings = common_ratings.loc[user_id]
    other_ratings = common_ratings.loc[other_user_id]

    if user_ratings.nunique() < 2 or other_ratings.nunique() < 2:
        return None

    similarity = user_ratings.corr(other_ratings)

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
    min_movie_ratings: int = 20,
) -> pd.DataFrame:
    """
    Recommend the top-N unseen movies for a user.
    User similarities are calculated only once.
    """

    result_columns = [
        "movie_id",
        "title",
        "predicted_rating",
        "raw_prediction",
    ]

    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=result_columns)

    required_columns = {"movieId", "title"}

    if not required_columns.issubset(movies.columns):
        raise ValueError("Movies DataFrame must contain movieId and title.")

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    target_user_mean = user_item_matrix.loc[user_id].mean()
    user_means = user_item_matrix.mean(axis=1)

    # Calculate similarities only once.
    similarities = {}

    for other_user_id in user_item_matrix.index:
        if other_user_id == user_id:
            continue

        similarity = calculate_user_similarity(
            user_item_matrix,
            user_id,
            other_user_id,
            min_common_ratings=min_common_ratings,
        )

        if similarity is not None and similarity > 0:
            similarities[other_user_id] = similarity

    # Find movies the user has not rated.
    unseen_movie_ids = user_item_matrix.columns[
        user_item_matrix.loc[user_id].isna()
    ]

    movie_titles = (
        movies.drop_duplicates(subset=["movieId"])
        .set_index("movieId")["title"]
    )

    recommendations = []

    for movie_id in unseen_movie_ids:

        if movie_id not in movie_titles.index:
            continue

        movie_ratings = user_item_matrix[movie_id].dropna()
        if len(movie_ratings) < min_movie_ratings:
            continue

        weighted_sum = 0.0
        similarity_sum = 0.0

        for other_user_id, rating in movie_ratings.items():

            similarity = similarities.get(other_user_id)

            if similarity is None:
                continue

            weighted_sum += similarity * (
                rating - user_means.loc[other_user_id]
            )
            similarity_sum += similarity

        if similarity_sum == 0:
            continue

        raw_prediction = (
                target_user_mean + weighted_sum / similarity_sum
        )

        predicted_rating = max(0.5, min(5.0, raw_prediction))

        recommendations.append({
            "movie_id": movie_id,
            "title": movie_titles.loc[movie_id],
            "predicted_rating": float(predicted_rating),
            "raw_prediction": float(raw_prediction),
        })
    recommendations_df = pd.DataFrame(
        recommendations,
        columns=result_columns,
    )
    return (
        recommendations_df.sort_values(
            "raw_prediction", ascending=False
        )
            .head(n)
            .drop(columns=["raw_prediction"])
            .reset_index(drop=True)
    )

