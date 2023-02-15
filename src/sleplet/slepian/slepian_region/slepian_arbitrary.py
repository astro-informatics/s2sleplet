from concurrent.futures import ThreadPoolExecutor
from dataclasses import KW_ONLY
from pathlib import Path

import numpy as np
import pyssht as ssht
from numpy import linalg as LA
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from sleplet.utils.config import settings
from sleplet.utils.harmonic_methods import create_spherical_harmonic, invert_flm_boosted
from sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_region_sphere,
)
from sleplet.utils.logger import logger
from sleplet.utils.mask_methods import create_mask_region
from sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)
from sleplet.utils.region import Region
from sleplet.utils.slepian_arbitrary_methods import clean_evals_and_evecs
from sleplet.utils.validation import Validation
from sleplet.utils.vars import L_MAX_DEFAULT, L_MIN_DEFAULT

_file_location = Path(__file__).resolve()
_slepian_path = _file_location.parents[2] / "data" / "slepian"


@dataclass(config=Validation)
class SlepianArbitrary(SlepianFunctions):
    mask_name: str
    _: KW_ONLY
    L_max: int = settings["L_MAX"]
    L_min: int = settings["L_MIN"]

    def __post_init_post_parse__(self) -> None:
        self.resolution = settings["SAMPLES"] * self.L
        super().__post_init_post_parse__()

    def _create_fn_name(self) -> str:
        return f"slepian_{self.mask_name}"

    def _create_region(self) -> Region:
        return Region(mask_name=self.mask_name)

    def _create_mask(self) -> npt.NDArray[np.float_]:
        return create_mask_region(self.resolution, self.region)

    def _calculate_area(self) -> float:
        self.weight = calc_integration_weight(self.resolution)
        return (self.mask * self.weight).sum()

    def _create_matrix_location(self) -> Path:
        return (
            _slepian_path / "eigensolutions" / f"D_{self.mask_name}_L{self.L}_N{self.N}"
        )

    def _solve_eigenproblem(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        eval_loc = self.matrix_location / "eigenvalues.npy"
        evec_loc = self.matrix_location / "eigenvectors.npy"
        if not eval_loc.exists() or not evec_loc.exists():
            return self._solve_D_matrix(eval_loc, evec_loc)

        logger.info("binaries found - loading...")
        return np.load(eval_loc), np.load(evec_loc)

    def _solve_D_matrix(self, eval_loc, evec_loc):
        D = self._create_D_matrix()

        # check whether the large job has been split up
        if (self.L_min != L_MIN_DEFAULT or self.L_max != self.L) and settings[
            "SAVE_MATRICES"
        ]:
            logger.info("large job has been used, saving intermediate matrix")
            inter_loc = self.matrix_location / f"D_min{self.L_min}_max{self.L_max}.npy"
            np.save(inter_loc, D)
            raise RuntimeError("Large job detected, exiting")

        # fill in remaining triangle section
        fill_upper_triangle_of_hermitian_matrix(D)

        # solve eigenproblem
        eigenvalues, eigenvectors = clean_evals_and_evecs(LA.eigh(D))
        if settings["SAVE_MATRICES"]:
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors[: self.N])
        return eigenvalues, eigenvectors

    def _create_D_matrix(self) -> npt.NDArray[np.complex_]:
        """
        computes the D matrix in parallel
        """
        # create dictionary for the integrals
        self._fields: dict[int, npt.NDArray[np.complex_ | np.float_]] = {}

        # initialise real and imaginary matrices
        D_r = np.zeros((self.L**2, self.L**2))
        D_i = np.zeros((self.L**2, self.L**2))

        D_r_ext, shm_r_ext = create_shared_memory_array(D_r)
        D_i_ext, shm_i_ext = create_shared_memory_array(D_i)

        def func(chunk: list[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            D_r_int, shm_r_int = attach_to_shared_memory_block(D_r, shm_r_ext)
            D_i_int, shm_i_int = attach_to_shared_memory_block(D_i, shm_i_ext)

            for i in chunk:
                logger.info(f"start ell: {i}")
                self._matrix_helper(D_r_int, D_i_int, i)
                logger.info(f"finish ell: {i}")

            free_shared_memory(shm_r_int, shm_i_int)

        # split up L range to maximise effiency
        chunks = split_arr_into_chunks(
            self.L_max**2, settings["NCPU"], arr_min=self.L_min**2
        )

        # initialise pool and apply function
        with ThreadPoolExecutor(max_workers=settings["NCPU"]) as e:
            e.map(func, chunks)

        # retrieve from parallel function
        D = D_r_ext + 1j * D_i_ext

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_r_ext, shm_i_ext)
        release_shared_memory(shm_r_ext, shm_i_ext)
        return D

    def _matrix_helper(
        self, D_r: npt.NDArray[np.float_], D_i: npt.NDArray[np.float_], i: int
    ) -> None:
        """
        used in both serial and parallel calculations

        the hack with splitting into real and imaginary parts
        is not required for the serial case but here for ease
        """
        # fill in diagonal components
        integral = self._integral(i, i)
        D_r[i][i] = integral.real
        D_i[i][i] = integral.imag
        _, m_i = ssht.ind2elm(i)

        for j in range(i + 1, D_r.shape[0]):
            ell_j, m_j = ssht.ind2elm(j)
            # if possible to use previous calculations
            if m_i == 0 and m_j != 0 and ell_j < self.L:
                # if positive m then use conjugate relation
                if m_j > 0:
                    integral = self._integral(j, i)
                    D_r[j][i] = integral.real
                    D_i[j][i] = integral.imag
                    k = ssht.elm2ind(ell_j, -m_j)
                    D_r[k][i] = (-1) ** m_j * D_r[j][i]
                    D_i[k][i] = (-1) ** (m_j + 1) * D_i[j][i]
            else:
                integral = self._integral(j, i)
                D_r[j][i] = integral.real
                D_i[j][i] = integral.imag

    def _integral(self, i: int, j: int) -> complex:
        """
        calculates the D integral between two spherical harmonics
        """
        if i not in self._fields:
            self._fields[i] = invert_flm_boosted(
                create_spherical_harmonic(self.L, i), self.L, self.resolution
            )
        if j not in self._fields:
            self._fields[j] = invert_flm_boosted(
                create_spherical_harmonic(self.L, j), self.L, self.resolution
            )
        return integrate_region_sphere(
            self.mask, self.weight, self._fields[i], self._fields[j].conj()
        )

    @validator("L_max")
    def check_L_max(cls, v, values):
        if v > values["L"]:
            raise ValueError(f"L_max cannot be greater than L: {values['L']}")
        if not isinstance(v, int):
            raise TypeError("L_max must be an integer")
        return v if v != L_MAX_DEFAULT else values["L"]

    @validator("L_min")
    def check_L_min(cls, v):
        if v < 0:
            raise ValueError("L_min cannot be negative")
        if not isinstance(v, int):
            raise TypeError("L_min must be an integer")
        return v
