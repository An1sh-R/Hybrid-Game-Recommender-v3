"""Typed settings loaded from params.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class FilterParams:
    min_user_reviews: int
    max_user_reviews: int
    min_game_reviews: int


@dataclass
class SplitParams:
    val_users: int
    test_users: int
    n_seeds: int
    random_seed: int


@dataclass
class CFParams:
    confidence_scale: float
    shrinkage: float
    top_k: int
    chunk_size: int


@dataclass
class ContentParams:
    tag_weight: float
    top_k: int
    chunk_size: int
    embedding_model: str
    batch_size: int


@dataclass
class HybridParams:
    alpha_max: float
    popularity_halfpoint: float


@dataclass
class EvaluateParams:
    k: int
    alpha_max_grid: list[float]
    halfpoint_grid: list[float]


@dataclass
class Params:
    filter: FilterParams
    split: SplitParams
    cf: CFParams
    content: ContentParams
    hybrid: HybridParams
    evaluate: EvaluateParams


def load_params(path: Path) -> Params:
    """Load params.yaml into typed dataclasses.

    A misspelled or missing key raises TypeError straight away,
    because each dataclass only accepts its exact field names.
    """
    with path.open() as file:
        raw = yaml.safe_load(file)

    params = Params(
        filter=FilterParams(**raw["filter"]),
        split=SplitParams(**raw["split"]),
        cf=CFParams(**raw["cf"]),
        content=ContentParams(**raw["content"]),
        hybrid=HybridParams(**raw["hybrid"]),
        evaluate=EvaluateParams(**raw["evaluate"]),
    )
    return params
