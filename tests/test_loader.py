import pandas as pd

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