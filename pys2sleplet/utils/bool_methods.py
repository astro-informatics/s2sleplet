import numpy as np

from pys2sleplet.utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


def is_polar_cap(
    phi_min: float, phi_max: float, theta_min: float, theta_max: float
) -> bool:
    """
    circular mask at the north pole
    """
    return bool(
        phi_min == PHI_MIN_DEFAULT
        and phi_max == PHI_MAX_DEFAULT
        and theta_min == THETA_MIN_DEFAULT
        and theta_max != THETA_MAX_DEFAULT
    )


def is_limited_lat_lon(
    phi_min: float, phi_max: float, theta_min: float, theta_max: float
) -> bool:
    """
    a region defined by angles, just need one to not be the default
    """
    return bool(
        not is_polar_cap(phi_min, phi_max, theta_min, theta_max)
        and phi_min != PHI_MIN_DEFAULT
        or phi_max != PHI_MAX_DEFAULT
        or theta_min != THETA_MIN_DEFAULT
    )


def is_small_polar_cap(theta_max: float) -> bool:
    """
    assuming it is a polar cap small defined for visualisation purposes
    """
    return bool(theta_max <= np.pi / 4)
