from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mask_methods import ensure_masked_flm_bandlimited
from pys2sleplet.utils.noise import create_noise
from pys2sleplet.utils.plot_methods import calc_nearest_grid_point, calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.string_methods import filename_angle

_file_location = Path(__file__).resolve()


@dataclass  # type:ignore
class Functions:
    L: int
    extra_args: Optional[List[int]]
    region: Optional[Region]
    noise: bool
    _annotations: List[Dict] = field(default_factory=list, init=False, repr=False)
    _extra_args: Optional[List[int]] = field(default=None, init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _multipole: np.ndarray = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _reality: bool = field(default=False, init=False, repr=False)
    _region: Region = field(default=None, init=False, repr=False)
    _noise: bool = field(default=False, init=False, repr=False)
    _resolution: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.resolution = calc_plot_resolution(self.L)
        self._setup_args()
        self._create_name()
        self._create_annotations()
        self._set_spin()
        self._set_reality()
        self._create_flm()
        self._add_region_to_name()
        self._add_noise_to_signal()

    def rotate(
        self,
        alpha_pi_fraction: float,
        beta_pi_fraction: float,
        gamma_pi_fraction: float = 0,
    ) -> np.ndarray:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        # angles such that rotation and translation are equal
        alpha, beta = calc_nearest_grid_point(
            self.L, alpha_pi_fraction, beta_pi_fraction
        )
        gamma = gamma_pi_fraction * np.pi

        # rotate flms
        multipole = ssht.rotate_flms(self.multipole, alpha, beta, gamma, self.L)
        return multipole

    def translate(
        self, alpha_pi_fraction: float, beta_pi_fraction: float
    ) -> np.ndarray:
        """
        translates given flm on the sphere by alpha/beta
        """
        # angles such that rotation and translation are equal
        alpha, beta = calc_nearest_grid_point(
            self.L, alpha_pi_fraction, beta_pi_fraction
        )

        # numpy binary filename
        filename = (
            _file_location.parents[1]
            / "data"
            / "trans_dirac"
            / (
                f"trans_dd_L{self.L}_"
                f"{filename_angle(alpha_pi_fraction,beta_pi_fraction)}.npy"
            )
        )

        # check if file of translated dirac delta already
        # exists otherwise calculate translated dirac delta
        if filename.exists():
            glm = np.load(filename)
        else:
            glm = ssht.create_ylm(beta, alpha, self.L).conj()
            glm = glm.reshape(glm.size)

            # save to speed up for future
            if settings.SAVE_MATRICES:
                np.save(filename, glm)

        # convolve with flm
        if self.name == "dirac_delta":
            multipole = glm
        else:
            multipole = self.convolve(self.multipole, glm)
        return multipole

    def convolve(self, flm: np.ndarray, glm: np.ndarray) -> np.ndarray:
        """
        computes the sifting convolution of two arrays
        """
        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False

        return flm * glm.conj()

    def _add_region_to_name(self) -> None:
        """
        adds region to the name if present if not a Slepian function
        """
        if self.region is not None and "slepian" not in self.name:
            self.name += f"_{self.region.name_ending}"

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise:
            nlm = create_noise(self.L, self.multipole)
            self.multipole += nlm

    @property
    def annotations(self) -> List[Dict]:
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: List[Dict]) -> None:
        self._annotations = annotations

    @property  # type:ignore
    def extra_args(self) -> Optional[List[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[List[int]]) -> None:
        if isinstance(extra_args, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            extra_args = Functions._extra_args
        self._extra_args = extra_args

    @property  # type:ignore
    def L(self) -> int:
        return self._L

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @property
    def multipole(self) -> np.ndarray:
        return self._multipole

    @multipole.setter
    def multipole(self, multipole: np.ndarray) -> None:
        if self.region is not None and all(
            x not in self.name for x in {"slepian", "south_america"}
        ):
            multipole = ensure_masked_flm_bandlimited(
                multipole, self.L, self.region, self.reality, self.spin
            )
        self._multipole = multipole

    @property
    def name(self) -> np.ndarray:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type:ignore
    def noise(self) -> bool:
        return self._noise

    @noise.setter
    def noise(self, noise: bool) -> None:
        if isinstance(noise, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            noise = Functions._noise
        self._noise = noise

    @property
    def reality(self) -> bool:
        return self._reality

    @reality.setter
    def reality(self, reality: bool) -> None:
        self._reality = reality

    @property  # type:ignore
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        if isinstance(region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            region = Functions._region
        self._region = region

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int) -> None:
        self._resolution = resolution

    @property
    def spin(self) -> bool:
        return self._spin

    @spin.setter
    def spin(self, spin: bool) -> None:
        self._spin = spin

    @abstractmethod
    def _create_annotations(self) -> None:
        """
        creates the annotations for the plot
        """
        raise NotImplementedError

    @abstractmethod
    def _create_flm(self) -> None:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        """
        creates the name of the function
        """
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> None:
        """
        sets the reality flag to speed up computations
        """
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> None:
        """
        sets the spin value in computations
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        """
        initialises function specific args
        either default value or user input
        """
        raise NotImplementedError
