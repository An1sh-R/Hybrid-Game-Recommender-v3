"""One logging setup shared by pipeline scripts."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """Log INFO and above with a timestamp."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
