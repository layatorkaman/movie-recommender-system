import pandas as pd


def calculate_movie_means(
    ratings: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the average rating for each movie.
    """
    return ratings.groupby("movieId")["rating"].mean()


def predict_movie_mean(
    movie_means: pd.Series,
    movie_id: int,
    global_mean: float,
) -> float:
    """
    Predict a rating using the movie average.
    If the movie is unknown, use the global average.
    """
    prediction = movie_means.get(movie_id, global_mean)

    return float(prediction)