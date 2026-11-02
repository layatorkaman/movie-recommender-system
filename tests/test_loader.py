import pandas as pd
import pytest

from src.data.loader import load_ratings


def test_load_ratings(tmp_path):
    csv_path = tmp_path / "ratings.csv"

    data = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [10, 20],
        "rating": [4.0, 5.0],
    })

    data.to_csv(csv_path, index=False)

    loaded_data = load_ratings(csv_path)

    assert len(loaded_data) == 2
    assert list(loaded_data.columns) == ["userId", "movieId", "rating"]
    assert loaded_data.loc[0, "rating"] == 4.0


def test_load_ratings_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ratings(tmp_path / "missing.csv")


def test_load_ratings_missing_column(tmp_path):
    csv_path = tmp_path / "ratings.csv"

    pd.DataFrame({
        "userId": [1],
        "rating": [4.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_ratings(csv_path)


def test_load_ratings_empty_file(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("userId,movieId,rating\n")

    with pytest.raises(ValueError, match="empty"):
        load_ratings(csv_path)


def test_load_ratings_invalid_rating(tmp_path):
    csv_path = tmp_path / "ratings.csv"

    pd.DataFrame({
        "userId": [1],
        "movieId": [10],
        "rating": [6.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="between 0.5 and 5.0"):
        load_ratings(csv_path)


def test_load_ratings_missing_value(tmp_path):
    csv_path = tmp_path / "ratings.csv"

    pd.DataFrame({
        "userId": [1],
        "movieId": [10],
        "rating": [None],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_ratings(csv_path)

def test_load_ratings_duplicate_interactions(tmp_path):
    csv_path = tmp_path / "ratings.csv"

    pd.DataFrame({
        "userId": [1, 1],
        "movieId": [10, 10],
        "rating": [4.0, 5.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_ratings(csv_path)