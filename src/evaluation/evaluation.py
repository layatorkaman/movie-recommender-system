import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.recommenders.baseline import (
    calculate_movie_means,
    predict_movie_mean,
)
from src.recommenders.item_based import predict_rating_item_based
from src.recommenders.user_based import (
    build_user_item_matrix,
    predict_rating,
)


def validate_test_size(test_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")


def validate_max_test_samples(max_test_samples: int) -> None:
    if (
            not isinstance(max_test_samples, int)
            or isinstance(max_test_samples, bool)
            or max_test_samples <= 0
    ):
        raise ValueError("max_test_samples must be a positive integer.")


def validate_min_common_ratings(min_common_ratings: int) -> None:
    if (
            not isinstance(min_common_ratings, int)
            or isinstance(min_common_ratings, bool)
            or min_common_ratings <= 0
    ):
        raise ValueError("min_common_ratings must be a positive integer.")


def split_ratings(
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_test_size(test_size)

    train_data, test_data = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )

    return train_data, test_data


def calculate_metrics(
        actual_ratings: list[float],
        predicted_ratings: list[float],
) -> dict[str, float]:
    mae = mean_absolute_error(
        actual_ratings,
        predicted_ratings,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_ratings,
            predicted_ratings,
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def evaluate_user_based(
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        min_common_ratings: int = 3,
        max_test_samples: int = 500,
) -> dict[str, float | int]:
    validate_max_test_samples(max_test_samples)
    validate_min_common_ratings(min_common_ratings)
    train_data, test_data = split_ratings(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )

    user_item_matrix = build_user_item_matrix(train_data)

    # برای جلوگیری از اجرای بسیار طولانی، فعلاً ۵۰۰ نمونه را ارزیابی می‌کنیم.
    test_sample = test_data.sample(
        n=min(max_test_samples, len(test_data)),
        random_state=random_state,
    )

    actual_ratings = []
    predicted_ratings = []

    for row in test_sample.itertuples(index=False):

        prediction = predict_rating(
            user_item_matrix,
            user_id=row.userId,
            movie_id=row.movieId,
            min_common_ratings=min_common_ratings,
        )

        if prediction is None:
            continue

        actual_ratings.append(float(row.rating))
        predicted_ratings.append(float(prediction))

    if not actual_ratings:
        raise ValueError("No ratings could be predicted.")

    metrics = calculate_metrics(
        actual_ratings,
        predicted_ratings,
    )

    metrics["evaluated_samples"] = len(actual_ratings)
    metrics["skipped_samples"] = (
        len(test_sample) - len(actual_ratings)
    )

    return metrics


def evaluate_baseline(
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        max_test_samples: int = 500,
) -> dict[str, float | int]:
    validate_max_test_samples(max_test_samples)
    train_data, test_data = split_ratings(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )

    movie_means = calculate_movie_means(train_data)
    global_mean = float(train_data["rating"].mean())

    test_sample = test_data.sample(
        n=min(max_test_samples, len(test_data)),
        random_state=random_state,
    )

    actual_ratings = []
    predicted_ratings = []

    for row in test_sample.itertuples(index=False):
        prediction = predict_movie_mean(
            movie_means,
            movie_id=row.movieId,
            global_mean=global_mean,
        )

        actual_ratings.append(float(row.rating))
        predicted_ratings.append(prediction)

    metrics = calculate_metrics(
        actual_ratings,
        predicted_ratings,
    )

    metrics["evaluated_samples"] = len(actual_ratings)
    return metrics


def evaluate_model_comparison(
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        min_common_ratings: int = 3,
        max_test_samples: int = 500,
) -> dict:
    validate_max_test_samples(max_test_samples)
    validate_min_common_ratings(min_common_ratings)
    train_data, test_data = split_ratings(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )

    user_item_matrix = build_user_item_matrix(train_data)

    movie_means = calculate_movie_means(train_data)
    global_mean = float(train_data["rating"].mean())

    test_sample = test_data.sample(
        n=min(max_test_samples, len(test_data)),
        random_state=random_state,
    )

    # Store predictions for each model independently.
    user_predictions = {}
    item_predictions = {}

    # Store predictions on the common evaluation subset.
    common_actual = []
    common_user_predictions = []
    common_item_predictions = []
    common_baseline_predictions = []

    user_start = time.perf_counter()

    for row in test_sample.itertuples(index=False):

        prediction = predict_rating(
            user_item_matrix,
            user_id=row.userId,
            movie_id=row.movieId,
            min_common_ratings=min_common_ratings,
        )

        if prediction is not None:
            user_predictions[(row.userId, row.movieId)] = float(prediction)

    user_time = time.perf_counter() - user_start

    item_start = time.perf_counter()

    for row in test_sample.itertuples(index=False):

        prediction = predict_rating_item_based(
            user_item_matrix,
            user_id=row.userId,
            item_id=row.movieId,
            min_common_ratings=min_common_ratings,
        )

        if prediction is not None:
            item_predictions[(row.userId, row.movieId)] = float(prediction)

    item_time = time.perf_counter() - item_start

    # Evaluate all models on exactly the same successful samples.
    for row in test_sample.itertuples(index=False):

        key = (row.userId, row.movieId)

        if key not in user_predictions or key not in item_predictions:
            continue

        baseline_prediction = predict_movie_mean(
            movie_means,
            movie_id=row.movieId,
            global_mean=global_mean,
        )

        common_actual.append(float(row.rating))
        common_user_predictions.append(user_predictions[key])
        common_item_predictions.append(item_predictions[key])
        common_baseline_predictions.append(float(baseline_prediction))

    if not common_actual:
        raise ValueError("No common samples could be evaluated.")

    user_metrics = calculate_metrics(
        common_actual,
        common_user_predictions,
    )

    item_metrics = calculate_metrics(
        common_actual,
        common_item_predictions,
    )

    baseline_metrics = calculate_metrics(
        common_actual,
        common_baseline_predictions,
    )

    total_samples = len(test_sample)

    return {
        "Baseline": baseline_metrics,
        "User-Based": user_metrics,
        "Item-Based": item_metrics,

        "coverage": {
            "Baseline": total_samples,
            "User-Based": len(user_predictions),
            "Item-Based": len(item_predictions),
        },

        "coverage_percentage": {
            "Baseline": 100.0,
            "User-Based": len(user_predictions) / total_samples * 100,
            "Item-Based": len(item_predictions) / total_samples * 100,
        },

        "execution_time": {
            "User-Based": user_time,
            "Item-Based": item_time,
        },

        "evaluated_samples": len(common_actual),
        "skipped_samples": total_samples - len(common_actual),
        "total_test_samples": total_samples,
    }
