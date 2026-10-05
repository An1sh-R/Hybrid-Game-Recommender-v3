from pathlib import Path

import pytest

from steamrec.config import load_params

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_params_reads_repo_params_file():
    params = load_params(REPO_ROOT / "params.yaml")

    assert params.filter.min_user_reviews == 5
    assert params.filter.max_user_reviews == 168
    assert params.cf.shrinkage == 10.0
    assert params.evaluate.alpha_max_grid == [0.3, 0.5, 0.7, 0.9]


def test_load_params_rejects_misspelled_key(tmp_path):
    text = (REPO_ROOT / "params.yaml").read_text()
    broken = text.replace("min_user_reviews", "min_user_revews")
    path = tmp_path / "params.yaml"
    path.write_text(broken)

    with pytest.raises(TypeError):
        load_params(path)
