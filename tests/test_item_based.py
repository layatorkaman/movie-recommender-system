import pandas as pd

from src.recommenders.item_based import calculate_item_similarity
import pytest

def old_predict_rating_item_based(
    user_item_matrix,
    user_id,
    item_id,
    min_common_ratings=3,
):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    user_ratings = user_ratings.drop(item_id, errors="ignore")

    weighted_sum = 0
    similarity_sum = 0

    for other_item_id, rating in user_ratings.items():
        similarity = calculate_item_similarity(
            user_item_matrix,
            item_id,
            other_item_id,
            min_common_ratings,
        )

        if similarity is not None and similarity > 0:
            weighted_sum += similarity * rating
            similarity_sum += similarity

    if similarity_sum == 0:
        return None

    prediction = weighted_sum / similarity_sum

    return float(max(0.5, min(5.0, prediction)))
def test_calculate_item_similarity():
    data = {
        10: [5, 4, 1, 5],
        20: [4, 3, 2, 5],
        30: [None, 5, 2, 4],
    }

    matrix = pd.DataFrame(
        data,
        index=[1, 2, 3, 4],
    )

    similarity = calculate_item_similarity(
        matrix,
        item_id=10,
        other_item_id=20,
    )

    assert similarity is not None
    assert similarity > 0
def test_calculate_item_similarity_not_enough_common_users():
    data = {
        10: [5, 4, None],
        20: [4, None, 2],
    }

    matrix = pd.DataFrame(
        data,
        index=[1, 2, 3],
    )

    similarity = calculate_item_similarity(
        matrix,
        item_id=10,
        other_item_id=20,
        min_common_ratings=3,
    )

    assert similarity is None

def test_calculate_item_similarity_with_constant_ratings():
    data = {
        10: [5, 5, 5, 5],
        20: [4, 3, 2, 1],
    }

    matrix = pd.DataFrame(data)

    similarity = calculate_item_similarity(
        matrix,
        item_id=10,
        other_item_id=20,
    )

    assert similarity is None


from src.recommenders.item_based import predict_rating_item_based


def test_predict_rating_item_based():
    data = {
        10: [5, 4, 1, 5],
        20: [4, 3, 2, 5],
        30: [1, 2, 5, 1],
        40: [5, 4, 2, 5],
    }

    matrix = pd.DataFrame(
        data,
        index=[1, 2, 3, 4],
    )

    prediction = predict_rating_item_based(
        matrix,
        user_id=1,
        item_id=40,
        min_common_ratings=3,
    )

    assert prediction is not None
    assert 0.5 <= prediction <= 5.0

from src.recommenders.item_based import recommend_movies_item_based


def test_recommend_movies_item_based():
    data = {
        10: [5, 4, 1, 5, 3],
        20: [4, 3, 2, 5, 4],
        30: [1, 2, 5, 1, 2],
        40: [5, 4, 2, 5, 3],
        50: [4, 5, 1, 4, 2],
    }

    matrix = pd.DataFrame(
        data,
        index=[1, 2, 3, 4, 5],
    )

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

    recommendations = recommend_movies_item_based(
        matrix,
        movies,
        user_id=1,
        n=2,
        min_common_ratings=3,
        min_movie_ratings=1,
    )

    assert len(recommendations) <= 2
    assert "movie_id" in recommendations.columns
    assert "title" in recommendations.columns
    assert "predicted_rating" in recommendations.columns

    assert 10 not in recommendations["movie_id"].values
    assert 20 not in recommendations["movie_id"].values
    assert 30 not in recommendations["movie_id"].values

    assert recommendations["predicted_rating"].is_monotonic_decreasing

def test_vectorized_prediction_matches_old_method():
    data = {
        10: [5, 4, 1, 5],
        20: [4, 3, 2, 5],
        30: [1, 2, 5, 1],
        40: [5, 4, 2, 5],
    }

    matrix = pd.DataFrame(
        data,
        index=[1, 2, 3, 4],
    )

    old_prediction = old_predict_rating_item_based(
        matrix,
        user_id=1,
        item_id=40,
        min_common_ratings=3,
    )

    new_prediction = predict_rating_item_based(
        matrix,
        user_id=1,
        item_id=40,
        min_common_ratings=3,
    )

    assert old_prediction is not None
    assert new_prediction is not None

    assert new_prediction == pytest.approx(
        old_prediction,
        abs=1e-10,
    )