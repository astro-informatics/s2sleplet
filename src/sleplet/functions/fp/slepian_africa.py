import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.f_p
import sleplet.functions.flm.africa
import sleplet.region
import sleplet.slepian_methods


@dataclass(config=sleplet._validation.Validation)
class SlepianAfrica(sleplet.functions.f_p.F_P):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()
        if (
            isinstance(self.region, sleplet.region.Region)
            and self.region.name_ending != "africa"
        ):
            raise RuntimeError("Slepian region selected must be 'africa'")

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        a = sleplet.functions.flm.africa.Africa(self.L, smoothing=self.smoothing)
        return sleplet.slepian_methods.slepian_forward(
            self.L, self.slepian, flm=a.coefficients
        )

    def _create_name(self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
