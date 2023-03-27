from __future__ import annotations

import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max

from sleplet import logger
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.meshes.slepian_coefficients.mesh_slepian_field import MeshSlepianField
from sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets import (
    MeshSlepianWavelets,
)
from sleplet.utils.string_methods import filename_args, wavelet_ending
from sleplet.utils.validation import Validation
from sleplet.utils.wavelet_methods import slepian_wavelet_forward


@dataclass(config=Validation, kw_only=True)
class MeshSlepianWaveletCoefficients(MeshSlepianCoefficients):
    B: int = 3
    j_min: int = 2
    j: int | None = None

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        logger.info("start computing wavelet coefficients")
        self.wavelets, self.wavelet_coefficients = self._create_wavelet_coefficients()
        logger.info("finish computing wavelet coefficients")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelet_coefficients[jth]

    def _create_name(self) -> str:
        return (
            f"slepian_wavelet_coefficients_{self.mesh.name}"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelet_coefficients(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_ | np.float_]]:
        """
        computes wavelet coefficients in Slepian space
        """
        smw = MeshSlepianWavelets(self.mesh, B=self.B, j_min=self.j_min)
        smf = MeshSlepianField(self.mesh)
        wavelets = smw.wavelets
        wavelet_coefficients = slepian_wavelet_forward(
            smf.coefficients,
            wavelets,
            self.mesh_slepian.N,
        )
        return wavelets, wavelet_coefficients

    @validator("j")
    def check_j(cls, v, values):
        j_max = pys2let_j_max(
            values["B"], values["mesh"].mesh_eigenvalues.shape[0], values["j_min"]
        )
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - values["j_min"]:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - values['j_min'] + 1}"
            )
        return v
