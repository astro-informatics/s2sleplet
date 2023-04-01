import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max, wavelet_tiling

import sleplet
import sleplet._string_methods
import sleplet._validation
import sleplet.functions.flm


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class DirectionalSpinWavelets(sleplet.functions.flm.FLM):
    """TODO."""

    B: int = 3
    """TODO"""
    j_min: int = 2
    """TODO"""
    j: int | None = None
    """TODO"""
    N: int = 2
    """TODO"""
    spin: int = 0
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        sleplet.logger.info("start computing wavelets")
        self.wavelets = self._create_wavelets()
        sleplet.logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelets[jth]

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.B, 'B')}"
            f"{sleplet._string_methods.filename_args(self.j_min, 'jmin')}"
            f"{sleplet._string_methods.filename_args(self.spin, 'spin')}"
            f"{sleplet._string_methods.filename_args(self.N, 'N')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> bool:
        return self.j is None or self.spin == 0

    def _set_spin(self) -> int:
        return self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 5
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.N, self.j = self.extra_args

    def _create_wavelets(self) -> npt.NDArray[np.complex_]:
        """Compute all wavelets."""
        phi_l, psi_lm = wavelet_tiling(self.B, self.L, self.N, self.j_min, self.spin)
        wavelets = np.zeros((psi_lm.shape[1] + 1, self.L**2), dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            wavelets[0, ind] = phi_l[ell]
        wavelets[1:] = psi_lm.T
        return wavelets

    @validator("j")
    def _check_j(cls, v, values):
        j_max = pys2let_j_max(values["B"], values["L"], values["j_min"])
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - values["j_min"]:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - values['j_min'] + 1}",
            )
        return v