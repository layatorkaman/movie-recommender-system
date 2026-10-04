from pathlib import Path

import pandas as pd


def load_ratings(path: str | Path) -> pd.DataFrame:
    """
    Load MovieLens ratings from a CSV file.
    """
    return pd.read_csv(path)