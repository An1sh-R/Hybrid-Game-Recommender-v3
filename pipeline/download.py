"""Pipeline stage: download the Kaggle dataset into data/raw/.

kagglehub saves downloads in its own cache folder, so we copy the files
into data/raw/ where DVC (and the rest of the pipeline) can find them.
"""

import logging
import shutil
from pathlib import Path

import kagglehub

DATASET = "antonkozyriev/game-recommendations-on-steam"
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

logger = logging.getLogger(__name__)


def download_dataset(dataset: str, raw_dir: Path) -> None:
    """Download a Kaggle dataset and copy its files into raw_dir."""
    cache_dir = Path(kagglehub.dataset_download(dataset))
    logger.info("downloaded to kagglehub cache: %s", cache_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in cache_dir.iterdir():
        target = raw_dir / source.name
        shutil.copy2(source, target)
        size_mb = target.stat().st_size / 1_000_000
        logger.info("copied %s (%.1f MB)", target.name, size_mb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_dataset(DATASET, RAW_DIR)
