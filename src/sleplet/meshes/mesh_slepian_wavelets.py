import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max

import sleplet
import sleplet._string_methods
import sleplet._validation
import sleplet.meshes.mesh_slepian_coefficients
import sleplet.wavelet_methods


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class MeshSlepianWavelets(
    sleplet.meshes.mesh_slepian_coefficients.MeshSlepianCoefficients,
):
    """TODO."""

    B: int = 3
    """TODO"""
    j_min: int = 2
    """TODO"""
    j: int | None = None
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
            f"slepian_wavelets_{self.mesh.name}"
            f"{sleplet._string_methods.filename_args(self.B, 'B')}"
            f"{sleplet._string_methods.filename_args(self.j_min, 'jmin')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelets(self) -> npt.NDArray[np.float_]:
        """Creates the Slepian wavelets of the mesh."""
        return sleplet.wavelet_methods.create_kappas(
            self.mesh.mesh_eigenvalues.shape[0],
            self.B,
            self.j_min,
        )

    @validator("j")
    def _check_j(cls, v, values) -> int | None:
        j_max = pys2let_j_max(
            values["B"],
            values["mesh"].mesh_eigenvalues.shape[0],
            values["j_min"],
        )
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - values["j_min"]:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - values['j_min'] + 1}",
            )
        return v