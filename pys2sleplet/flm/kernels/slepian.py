from dataclasses import dataclass, field

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import choose_slepian_method


@dataclass
class Slepian(Functions):
    rank: int
    slepian: SlepianFunctions
    _rank: int = field(default=0, init=False, repr=False)
    _slepian: SlepianFunctions = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # create default region from config dict
        region = Region()
        self.slepian = choose_slepian_method(self.L, region)
        self._validate_rank()
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_name(self) -> None:
        self.name = f"{self.slepian.name}_rank{self.rank}"

    def _create_flm(self) -> None:
        self.multipole = self.slepian.eigenvectors[self.rank]
        logger.info(f"Eigenvalue {self.rank}: {self.slepian.eigenvalues[self.rank]:e}")

    def _set_reality(self) -> None:
        self.reality = False

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            self.rank = self.extra_args[0]

    def _validate_rank(self) -> None:
        """
        checks the requested rank is valid
        """
        if self.extra_args is not None:
            limit = self.slepian.eigenvectors.shape[0]
            if self.extra_args[0] >= limit:
                raise ValueError(f"rank should be less than or equal to {limit}")

    @property  # type: ignore
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        if isinstance(rank, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            rank = Slepian._rank
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        self._rank = rank
        logger.info(f"rank={self.rank}")

    @property  # type: ignore
    def slepian(self) -> SlepianFunctions:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: SlepianFunctions) -> None:
        self._slepian = slepian
