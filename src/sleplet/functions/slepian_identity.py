import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.fp


@dataclass(config=sleplet._validation.Validation)
class SlepianIdentity(sleplet.functions.fp.FP):
    """TODO."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return np.ones(self.L**2, dtype=np.complex_)

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region.name_ending}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )