

import numpy as np
import pandas as pd

def calculate_item_similarity(
    user_item_matrix: pd.DataFrame,
    item_id: int,
    other_item_id: int,
    min_common_ratings: int = 3,
) -> float | None:

    common_ratings = user_item_matrix[
        [item_id, other_item_id]
    ].dropna()

    if len(common_ratings) < min_common_ratings:
        return None

    item_ratings = common_ratings[item_id]
    other_item_ratings = common_ratings[other_item_id]

    if item_ratings.nunique() < 2 or other_item_ratings.nunique() < 2:
        return None

    similarity = item_ratings.corr(other_item_ratings)

    if pd.isna(similarity):
        return None

    return float(similarity)

def predict_rating_item_based(
    user_item_matrix: pd.DataFrame,
    user_id: int,
    item_id: int,
    min_common_ratings: int = 3,
) -> float | None:

    if user_id not in user_item_matrix.index:
        return None

    if item_id not in user_item_matrix.columns:
        return None

    # Get movies rated by the user.
    user_ratings = user_item_matrix.loc[user_id].dropna()

    # Remove the target movie if already rated.
    user_ratings = user_ratings.drop(
        labels=[item_id],
        errors="ignore",
    )

    if user_ratings.empty:
        return None

    other_item_ids = user_ratings.index
    ratings = user_ratings.to_numpy(dtype=float)

    # Get ratings of the target movie and other movies.
    target = user_item_matrix[item_id].to_numpy(dtype=float)

    other = user_item_matrix[
        other_item_ids
    ].to_numpy(dtype=float)

    # Find users who rated both the target and each other movie.
    valid = (
        np.isfinite(other)
        & np.isfinite(target[:, np.newaxis])
    )

    counts = valid.sum(axis=0)

    target_values = np.where(
        valid,
        target[:, np.newaxis],
        0.0,
    )

    other_values = np.where(
        valid,
        other,
        0.0,
    )

    # Calculate the required sums for Pearson correlation.
    sum_target = target_values.sum(axis=0)
    sum_other = other_values.sum(axis=0)

    sum_target_sq = (target_values ** 2).sum(axis=0)
    sum_other_sq = (other_values ** 2).sum(axis=0)

    sum_product = (
        target_values * other_values
    ).sum(axis=0)

    # Calculate Pearson correlation for all rated movies.
    safe_counts = np.maximum(counts, 1)

    covariance = (
        sum_product
        - (sum_target * sum_other) / safe_counts
    )

    variance_target = (
        sum_target_sq
        - (sum_target ** 2) / safe_counts
    )

    variance_other = (
        sum_other_sq
        - (sum_other ** 2) / safe_counts
    )

    denominator = np.sqrt(
        np.maximum(variance_target, 0)
        * np.maximum(variance_other, 0)
    )

    similarities = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=(
            (counts >= min_common_ratings)
            & (denominator > 0)
        ),
    )

    # Use only positive similarities.
    positive = similarities > 0

    if not np.any(positive):
        return None

    selected_similarities = similarities[positive]
    selected_ratings = ratings[positive]

    prediction = (
        np.dot(selected_similarities, selected_ratings)
        / selected_similarities.sum()
    )

    return float(max(0.5, min(5.0, prediction)))

def recommend_movies_item_based(
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 10,
    min_common_ratings: int = 3,
    min_movie_ratings: int = 20,
) -> pd.DataFrame:
    """
    Recommend the top-N unseen movies using Item-Based CF.
    """

    result_columns = [
        "movie_id",
        "title",
        "predicted_rating",
    ]

    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=result_columns)

    required_columns = {"movieId", "title"}

    if not required_columns.issubset(movies.columns):
        raise ValueError(
            "Movies DataFrame must contain movieId and title."
        )

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    if min_movie_ratings < 1:
        raise ValueError("min_movie_ratings must be positive.")

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

        prediction = predict_rating_item_based(
            user_item_matrix,
            user_id=user_id,
            item_id=movie_id,
            min_common_ratings=min_common_ratings,
        )

        if prediction is None:
            continue

        recommendations.append({
            "movie_id": movie_id,
            "title": movie_titles.loc[movie_id],
            "predicted_rating": prediction,
        })

    recommendations_df = pd.DataFrame(
        recommendations,
        columns=result_columns,
    )

    return (
        recommendations_df
        .sort_values(
            "predicted_rating",
            ascending=False,
        )
        .head(n)
        .reset_index(drop=True)
    )
