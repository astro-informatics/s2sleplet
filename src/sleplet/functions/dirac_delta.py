"""Contains the `DiracDelta` class."""
import numpy as np
import numpy.typing as npt
import pydantic

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class DiracDelta(Flm):
    """Create a Dirac delta."""

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        return flm

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
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise AttributeError(msg)
