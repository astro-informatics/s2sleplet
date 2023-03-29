import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet._validation import Validation
from sleplet.data.create_earth_flm import create_flm
from sleplet.functions.f_lm import F_LM
from sleplet.string_methods import _convert_camel_case_to_snake_case


@dataclass(config=Validation)
class Earth(F_LM):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return create_flm(self.L, smoothing=self.smoothing)

    def _create_name(self) -> str:
        return _convert_camel_case_to_snake_case(self.__class__.__name__)

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
