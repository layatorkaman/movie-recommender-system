from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"userId", "movieId", "rating"}


def load_ratings(path: str | Path) -> pd.DataFrame:
    """
    Load and validate MovieLens ratings from a CSV file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found: {path}")

    ratings = pd.read_csv(path)

    if ratings.empty:
        raise ValueError("Ratings file is empty.")

    missing_columns = REQUIRED_COLUMNS - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    required_data = ratings[list(REQUIRED_COLUMNS)]

    if required_data.isna().any().any():
        raise ValueError("Required columns contain missing values.")

    if not pd.api.types.is_numeric_dtype(ratings["userId"]):
        raise ValueError("userId must be numeric.")

    if not pd.api.types.is_numeric_dtype(ratings["movieId"]):
        raise ValueError("movieId must be numeric.")

    if not pd.api.types.is_numeric_dtype(ratings["rating"]):
        raise ValueError("rating must be numeric.")

    if (ratings["userId"] <= 0).any():
        raise ValueError("userId must be positive.")

    if (ratings["movieId"] <= 0).any():
        raise ValueError("movieId must be positive.")

    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0.")

    if not (ratings["rating"] * 2).mod(1).eq(0).all():
        raise ValueError("Ratings must use increments of 0.5.")

    duplicate_count = ratings.duplicated(
        subset=["userId", "movieId"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate user-movie interactions."
        )

    return ratings