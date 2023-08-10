"""Contains the `Earth` class."""
import numpy as np
import pydantic
from numpy import typing as npt

import sleplet._data.create_earth_flm
import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class Earth(Flm):
    """Creates the topographic map of the Earth."""

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return sleplet._data.create_earth_flm.create_flm(
            self.L,
            smoothing=self.smoothing,
        )

    def _create_name(self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__,
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )
