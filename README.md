# Hybrid Game Recommender

A hybrid Steam game recommender combining **item-to-item collaborative filtering** ("people who liked X also liked Y") with **content-based filtering** (tags + description embeddings), built with MLOps practices: DVC pipelines, CI/CD, Docker and cloud deployment.

Pick 3 games you like → get recommendations, each explained ("Because you liked X") with a breakdown of how much came from collaborative vs content signals.

> 🚧 Work in progress. See [`docs/design-decisions.md`](docs/design-decisions.md) for the design and [`docs/coding-standards.md`](docs/coding-standards.md) for how the code is written.

## Data

[Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (Kaggle): ~41M user reviews, ~50k games with tags and descriptions.

## Setup

```bash
uv sync                          # create .venv and install dependencies
uv run python pipeline/download.py   # download the dataset into data/raw/
```

*Not affiliated with Valve or Steam.*
