"""Load the raw Kaggle files with explicit columns and small dtypes."""

from pathlib import Path

import pandas as pd

GAMES_DTYPES = {
    "app_id": "int32",
    "title": "string",
    "rating": "category",
    "positive_ratio": "int8",
    "user_reviews": "int32",
    "price_final": "float32",
    "win": "bool",
    "mac": "bool",
    "linux": "bool",
    "steam_deck": "bool",
}

RECOMMENDATIONS_DTYPES = {
    "app_id": "int32",
    "user_id": "int32",
    "is_recommended": "bool",
    "hours": "float32",
}


def load_games(raw_dir: Path) -> pd.DataFrame:
    """Load games.csv: one row per game."""
    games = pd.read_csv(
        raw_dir / "games.csv",
        usecols=list(GAMES_DTYPES) + ["date_release"],
        dtype=GAMES_DTYPES,
        parse_dates=["date_release"],
    )
    return games


def load_recommendations(raw_dir: Path) -> pd.DataFrame:
    """Load recommendations.csv: one row per (user, game) review, ~41M rows.

    Only the columns the recommender needs are read, with small dtypes,
    which keeps memory to roughly 0.5 GB instead of several GB.
    """
    recommendations = pd.read_csv(
        raw_dir / "recommendations.csv",
        usecols=list(RECOMMENDATIONS_DTYPES),
        dtype=RECOMMENDATIONS_DTYPES,
    )
    return recommendations


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Load games_metadata.json: description and tag list per game (one JSON object per line)."""
    metadata = pd.read_json(raw_dir / "games_metadata.json", lines=True)
    metadata["app_id"] = metadata["app_id"].astype("int32")
    return metadata
