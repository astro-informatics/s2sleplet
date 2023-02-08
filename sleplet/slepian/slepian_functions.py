from abc import abstractmethod

import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.utils.logger import logger
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class SlepianFunctions:
    L: int

    def __post_init__(self) -> None:
        self.mask = self._create_mask()
        self.name = self._create_fn_name()
        area = self._calculate_area()
        self.N = round(area * self.L**2 / (4 * np.pi))
        logger.info(f"Shannon number N={self.N}")
        self._create_matrix_location()
        logger.info("start solving eigenproblem")
        self._solve_eigenproblem()
        logger.info("finished solving eigenproblem")

    @abstractmethod
    def _create_fn_name(self) -> str:
        """
        creates the name for plotting
        """
        raise NotImplementedError

    @abstractmethod
    def _create_mask(self) -> np.ndarray:
        """
        creates a mask of the region of interest
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_area(self) -> float:
        """
        calculates area of region
        """
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> None:
        """
        creates the name of the matrix binary
        """
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self) -> None:
        """
        solves the eigenproblem for the given function
        """
        raise NotImplementedError
