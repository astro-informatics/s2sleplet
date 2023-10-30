"""Contains the abstract `MeshSlepianCoefficients` class."""
import abc

import numpy as np
import numpy.typing as npt
import pydantic

import sleplet._validation
import sleplet.noise
from sleplet.meshes.mesh import Mesh
from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.meshes.mesh_slepian import MeshSlepian


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshSlepianCoefficients(MeshCoefficients):
    """Abstract parent class to handle Slepian coefficients on the mesh."""

    mesh_slepian: MeshSlepian = pydantic.Field(
        default=MeshSlepian(Mesh("bird")),
        init_var=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.mesh_slepian = sleplet.meshes.mesh_slepian.MeshSlepian(self.mesh)
        super().__post_init__()

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Add Gaussian white noise converted to Slepian space."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            n_p = sleplet.noise._create_slepian_mesh_noise(
                self.mesh_slepian,
                self.coefficients,
                self.noise,
            )
            snr = sleplet.noise.compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients = self.coefficients + n_p
            return unnoised_coefficients, snr
        return None, None

    @abc.abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
