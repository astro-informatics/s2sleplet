from typing import Optional

from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max

from sleplet.functions.f_lm import F_LM
from sleplet.functions.flm.earth import Earth
from sleplet.utils.logger import logger
from sleplet.utils.string_methods import (
    convert_camel_case_to_snake_case,
    filename_args,
    wavelet_ending,
)
from sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    create_axisymmetric_wavelets,
)


@dataclass(kw_only=True)
class AxisymmetricWaveletCoefficientsEarth(F_LM):
    B: int = 3
    j_min: int = 2
    j: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        logger.info("start computing wavelet coefficients")
        self._create_wavelet_coefficients()
        logger.info("finish computing wavelet coefficients")
        jth = 0 if self.j is None else self.j + 1
        self.coefficients = self.wavelet_coefficients[jth]

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelet_coefficients(self) -> None:
        """
        computes wavelet coefficients of the Earth
        """
        self.wavelets = create_axisymmetric_wavelets(self.L, self.B, self.j_min)
        self.earth = Earth(self.L, smoothing=self.smoothing)
        self.wavelet_coefficients = axisymmetric_wavelet_forward(
            self.L, self.earth.coefficients, self.wavelets
        )

    @validator("j")
    def check_j(cls, j: Optional[int]) -> Optional[int]:
        cls.j_max = pys2let_j_max(cls.B, cls.L, cls.j_min)
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > cls.j_max - cls.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {cls.j_max - cls.j_min + 1}"
            )
        return j
