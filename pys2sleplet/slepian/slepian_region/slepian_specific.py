from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


@dataclass  # type: ignore
class SlepianSpecific(SlepianFunctions):
    _order: int = field(default=0, init=False, repr=False)
    _phi_max: float = field(default=PHI_MAX_DEFAULT, init=False, repr=False)
    _phi_min: float = field(default=PHI_MIN_DEFAULT, init=False, repr=False)
    _theta_max: float = field(default=THETA_MAX_DEFAULT, init=False, repr=False)
    _theta_min: float = field(default=THETA_MIN_DEFAULT, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def phi_min(self) -> float:
        return self._phi_min

    @phi_min.setter
    def phi_min(self, phi_min: float) -> None:
        if np.rad2deg(phi_min) < PHI_MIN_DEFAULT:
            raise ValueError("phi_min cannot be negative")
        if np.rad2deg(phi_min) > PHI_MAX_DEFAULT:
            raise ValueError(f"phi_min cannot be greater than {PHI_MAX_DEFAULT}")
        self._phi_min = phi_min

    @property
    def phi_max(self) -> float:
        return self._phi_max

    @phi_max.setter
    def phi_max(self, phi_max: float) -> None:
        if np.rad2deg(phi_max) < PHI_MIN_DEFAULT:
            raise ValueError("phi_max cannot be negative")
        if np.rad2deg(phi_max) > PHI_MAX_DEFAULT:
            raise ValueError(f"phi_max cannot be greater than {PHI_MAX_DEFAULT}")
        self._phi_max = phi_max

    @property
    def theta_min(self) -> float:
        return self._theta_min

    @theta_min.setter
    def theta_min(self, theta_min: float) -> None:
        if np.rad2deg(theta_min) < THETA_MIN_DEFAULT:
            raise ValueError("theta_min cannot be negative")
        if np.rad2deg(theta_min) > THETA_MAX_DEFAULT:
            raise ValueError(f"theta_min cannot be greater than {THETA_MAX_DEFAULT}")
        self._theta_min = theta_min

    @property
    def theta_max(self) -> float:
        return self._theta_max

    @theta_max.setter
    def theta_max(self, theta_max: float) -> None:
        if np.rad2deg(theta_max) < THETA_MIN_DEFAULT:
            raise ValueError("theta_max cannot be negative")
        if np.rad2deg(theta_max) > THETA_MAX_DEFAULT:
            raise ValueError(f"theta_max cannot be greater than {THETA_MAX_DEFAULT}")
        self._theta_max = theta_max

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        if not isinstance(order, int):
            raise TypeError("order should be an integer")
        # check order is in correct range
        if abs(order) >= self.L:
            raise ValueError(f"Order magnitude should be less than {self.L}")
        self._order = order

    @abstractmethod
    def _create_mask(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _create_annotations(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def _create_fn_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
