import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.earth
import sleplet.functions.flm
import sleplet.noise


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class NoiseEarth(sleplet.functions.flm.FLM):
    """TODO."""

    SNR: float = 10
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        earth = sleplet.functions.earth.Earth(self.L, smoothing=self.smoothing)
        noise = sleplet.noise._create_noise(self.L, earth.coefficients, self.SNR)
        sleplet.noise.compute_snr(earth.coefficients, noise, "Harmonic")
        return noise

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]