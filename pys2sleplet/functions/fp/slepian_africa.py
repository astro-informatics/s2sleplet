from dataclasses import dataclass

from pys2sleplet.functions.f_p import F_P
from pys2sleplet.functions.flm.africa import Africa
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.string_methods import convert_camel_case_to_snake_case


@dataclass
class SlepianAfrica(F_P):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.region.name_ending != "africa":
            raise RuntimeError("Slepian region selected must be 'africa'")

    def _create_coefficients(self) -> None:
        a = Africa(self.L, smoothing=self.smoothing)
        self.coefficients = slepian_forward(self.L, self.slepian, flm=a.coefficients)

    def _create_name(self) -> None:
        self.name = convert_camel_case_to_snake_case(self.__class__.__name__)

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
