import logging
import os

import pooch
from platformdirs import user_data_path

_logger = logging.getLogger(__name__)


_ZENODO_DATA_DOI = "10.5281/zenodo.7767698"
_POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    base_url=f"doi:{_ZENODO_DATA_DOI}/",
    registry=None,
)
_POOCH.load_registry_from_doi()


def find_on_pooch_then_local(filename: str) -> os.PathLike | None:
    """Find a file on POOCH first and if not look in data folder."""
    if filename in _POOCH.registry:
        _logger.info(f"Found {filename} at https://doi.org/{_ZENODO_DATA_DOI}")
        return _POOCH.fetch(filename, progressbar=True)
    if (user_data_path() / filename).exists():
        _logger.info(f"Found {filename} at {user_data_path() / filename}")
        return user_data_path() / filename
    _logger.info(f"No {filename} found, calculating...")
    return None