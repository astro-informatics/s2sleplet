"""
API Documentation of the `SLEPLET` package.\n
[![PyPI](https://badge.fury.io/py/sleplet.svg)](https://pypi.org/project/sleplet)
"""

import logging

from sleplet._version import __version__

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] --- %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
