from dataclasses import dataclass

import numexpr as ne
import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions


@dataclass
class DiracDelta(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = ne.evaluate(f"sqrt((2*ell+1)/(4*{np.pi}))")
        self.multipole = flm

    def _create_name(self) -> None:
        self.name = "dirac_delta"

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
