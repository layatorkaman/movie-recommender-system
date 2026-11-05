
import pandas as pd

from src.recommenders.user_based import (
    build_user_item_matrix,
    calculate_user_similarity,
    predict_rating
)


def test_build_user_item_matrix():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 10],
        "rating": [4.0, 5.0, 3.0],
    })

    matrix = build_user_item_matrix(ratings)

    assert matrix.loc[1, 10] == 4.0
    assert matrix.loc[1, 20] == 5.0
    assert matrix.loc[2, 10] == 3.0
    assert pd.isna(matrix.loc[2, 20])


def test_similarity_returns_none_with_too_few_common_ratings():
    ratings = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [10, 10],
        "rating": [5.0, 5.0],
    })

    matrix = build_user_item_matrix(ratings)

    similarity = calculate_user_similarity(
        matrix,
        user_id=1,
        other_user_id=2,
        min_common_ratings=3,
    )

    assert similarity is None


def test_similarity_returns_pearson_correlation():
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [10, 20, 30, 10, 20, 30],
        "rating": [1.0, 2.0, 3.0, 5.0, 4.0, 3.0],
    })

    matrix = build_user_item_matrix(ratings)

    similarity = calculate_user_similarity(
        matrix,
        user_id=1,
        other_user_id=2,
        min_common_ratings=3,
    )

    assert similarity is not None
    assert abs(similarity - (-1.0)) < 1e-9
def test_predict_rating():
    ratings = pd.DataFrame({
        "userId": [
            1, 1, 1,
            2, 2, 2, 2,
        ],
        "movieId": [
            10, 20, 30,
            10, 20, 30, 40,
        ],
        "rating": [
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0, 4.0,
        ],
    })

    matrix = build_user_item_matrix(ratings)

    prediction = predict_rating(
        matrix,
        user_id=1,
        movie_id=40,
        min_common_ratings=3,
    )

    assert prediction == 3.5
def test_predict_rating_unknown_user():
    ratings = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [10, 10],
        "rating": [4.0, 5.0],
    })

    matrix = build_user_item_matrix(ratings)

    prediction = predict_rating(
        matrix,
        user_id=99,
        movie_id=10,
    )

    assert prediction is None


def test_predict_rating_unknown_movie():
    ratings = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [10, 10],
        "rating": [4.0, 5.0],
    })

    matrix = build_user_item_matrix(ratings)

    prediction = predict_rating(
        matrix,
        user_id=1,
        movie_id=99,
    )

    assert prediction is None


def test_predict_rating_without_positive_neighbors():
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 2],
        "movieId": [10, 20, 30, 10, 20, 30, 40],
        "rating": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 5.0],
    })

    matrix = build_user_item_matrix(ratings)

    prediction = predict_rating(
        matrix,
        user_id=1,
        movie_id=40,
        min_common_ratings=3,
    )

    assert prediction is None



def test_predict_rating_does_not_exceed_maximum():
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 2],
        "movieId": [10, 20, 30, 10, 20, 30, 40],
        "rating": [3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 5.0],
    })

    matrix = build_user_item_matrix(ratings)

    prediction = predict_rating(
        matrix,
        user_id=1,
        movie_id=40,
        min_common_ratings=3,
    )

    assert prediction == 5.0


def test_predict_rating_does_not_go_below_minimum():
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 2],
        "movieId": [10, 20, 30, 10, 20, 30, 40],
        "rating": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 0.5],
    })

    matrix = build_user_item_matrix(ratings)

    prediction = predict_rating(
        matrix,
        user_id=1,
        movie_id=40,
        min_common_ratings=3,
    )

    assert prediction == 0.5


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
        min_movie_ratings=1,

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
