"""
methods to help in creating plots
"""
from pathlib import Path

import numpy as np
import pyssht as ssht
from matplotlib import colors
from matplotlib import pyplot as plt
from numpy import typing as npt

import sleplet
import sleplet._mask_methods
import sleplet._vars
import sleplet.functions.coefficients
import sleplet.harmonic_methods
import sleplet.meshes
import sleplet.meshes.mesh_coefficients
import sleplet.meshes.mesh_slepian_coefficients
import sleplet.region
import sleplet.slepian_methods


def calc_plot_resolution(L: int) -> int:
    """
    calculate appropriate resolution for given L
    """
    res_dict = {1: 6, 2: 5, 3: 4, 7: 3, 9: 2, 10: 1}

    return next(
        (
            L * 2**exponent
            for log_bandlimit, exponent in res_dict.items()
            if 2**log_bandlimit > L
        ),
        L,
    )


def _convert_colourscale(
    cmap: colors, *, pl_entries: int = 255
) -> list[tuple[float, str]]:
    """
    converts cmocean colourscale to a plotly colourscale
    """
    h = 1 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append((k * h, f"rgb{(C[0], C[1], C[2])}"))

    return pl_colorscale


def _calc_nearest_grid_point(
    L: int, alpha_pi_fraction: float, beta_pi_fraction: float
) -> tuple[float, float]:
    """
    calculate nearest index of alpha/beta for translation
    this is due to calculating omega' through the pixel
    values - the translation needs to be at the same position
    as the rotation such that the difference error is small
    """
    thetas, phis = ssht.sample_positions(L, Method=sleplet._vars.SAMPLING_SCHEME)
    pix_j = np.abs(phis - alpha_pi_fraction * np.pi).argmin()
    pix_i = np.abs(thetas - beta_pi_fraction * np.pi).argmin()
    alpha, beta = phis[pix_j], thetas[pix_i]
    sleplet.logger.info(f"grid point: (alpha, beta)=({alpha:e}, {beta:e})")
    return alpha, beta


def save_plot(path: Path, name: str) -> None:
    """
    helper method to save plots
    """
    plt.tight_layout()
    for file_type in {"png", "pdf"}:
        filename = path / file_type / f"{name}.{file_type}"
        sleplet.logger.info(f"saving {filename}")
        plt.savefig(filename, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def find_max_amplitude(
    function: sleplet.functions.coefficients.Coefficients,
    *,
    plot_type: str = "real",
    upsample: bool = True,
) -> float:
    """
    for a given set of coefficients it finds the largest absolute value for a
    given plot type such that plots can have the same scale as the input
    """
    # compute inverse transform
    if hasattr(function, "slepian"):
        field = sleplet.slepian_methods.slepian_inverse(
            function.coefficients, function.L, function.slepian
        )
    else:
        field = ssht.inverse(
            function.coefficients, function.L, Method=sleplet._vars.SAMPLING_SCHEME
        )

    # find resolution of final plot for boosting if necessary
    resolution = calc_plot_resolution(function.L) if upsample else function.L

    # boost field to match final plot
    boosted_field = _boost_field(
        field,
        function.L,
        resolution,
        reality=function.reality,
        spin=function.spin,
        upsample=upsample,
    )

    # find maximum absolute value for given plot type
    return np.abs(_create_plot_type(boosted_field, plot_type)).max()


def _create_plot_type(
    field: npt.NDArray[np.complex_ | np.float_], plot_type: str
) -> npt.NDArray[np.float_]:
    """
    gets the given plot type of the field
    """
    sleplet.logger.info(f"plotting type: '{plot_type}'")
    plot_dict = {
        "abs": np.abs(field),
        "imag": field.imag,
        "real": field.real,
        "sum": field.real + field.imag,
    }
    return plot_dict[plot_type]


def _set_outside_region_to_minimum(
    f_plot: npt.NDArray[np.float_], L: int, region: sleplet.region.Region
) -> npt.NDArray[np.float_]:
    """
    for the Slepian region set the outisde area to negative infinity
    hence it is clear we are only interested in the coloured region
    """
    # create mask of interest
    mask = sleplet._mask_methods.create_mask_region(L, region)

    # adapt for closed plot
    _, n_phi = ssht.sample_shape(L, Method=sleplet._vars.SAMPLING_SCHEME)
    closed_mask = np.insert(mask, n_phi, mask[:, 0], axis=1)

    # set values outside mask to negative infinity
    return np.where(closed_mask, f_plot, sleplet._vars.SPHERE_UNSEEN)


def _normalise_function(
    f: npt.NDArray[np.float_], *, normalise: bool
) -> npt.NDArray[np.float_]:
    """
    normalise function between 0 and 1 for visualisation
    """
    if not normalise:
        return f
    elif (f == 0).all():
        # if all 0, set to 0
        return f + 0.5
    elif np.allclose(f, f.max()):
        # if all non-zero, set to 1
        return f / f.max()
    else:
        # scale from [0, 1]
        return (f - f.min()) / f.ptp()


def _boost_field(
    field: npt.NDArray[np.complex_ | np.float_],
    L: int,
    resolution: int,
    *,
    reality: bool,
    spin: int,
    upsample: bool,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    inverts and then boosts the field before plotting
    """
    if not upsample:
        return field
    flm = ssht.forward(
        field, L, Reality=reality, Spin=spin, Method=sleplet._vars.SAMPLING_SCHEME
    )
    return sleplet.harmonic_methods.invert_flm_boosted(
        flm, L, resolution, reality=reality, spin=spin
    )


def compute_amplitude_for_noisy_mesh_plots(
    f: sleplet.meshes.mesh_coefficients.MeshCoefficients,
) -> float | None:
    """
    for the noised plots fix the amplitude to the initial data
    """
    return (
        np.abs(_coefficients_to_field_mesh(f, f.unnoised_coefficients)).max()
        if f.unnoised_coefficients is not None
        else None
    )


def _coefficients_to_field_mesh(
    f: sleplet.meshes.mesh_coefficients.MeshCoefficients,
    coefficients: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    computes the field over the whole mesh from the harmonic/Slepian coefficients
    """
    return (
        sleplet.slepian_methods.slepian_mesh_inverse(f.mesh_slepian, coefficients)
        if isinstance(
            f, sleplet.meshes.mesh_slepian_coefficients.MeshSlepianCoefficients
        )
        else sleplet.harmonic_methods.mesh_inverse(f.mesh, coefficients)
    )


def compute_amplitude_for_noisy_sphere_plots(
    f: sleplet.functions.coefficients.Coefficients,
) -> float | None:
    """
    for the noised plots fix the amplitude to the initial data
    """
    return (
        np.abs(_coefficients_to_field_sphere(f, f.unnoised_coefficients)).max()
        if f.unnoised_coefficients is not None
        else None
    )


def _coefficients_to_field_sphere(
    f: sleplet.functions.coefficients.Coefficients,
    coefficients: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    computes the field over the samples from the harmonic/Slepian coefficients
    """
    return (
        sleplet.slepian_methods.slepian_inverse(coefficients, f.L, f.slepian)
        if hasattr(f, "slepian")
        else ssht.inverse(
            coefficients,
            f.L,
            Reality=f.reality,
            Spin=f.spin,
            Method=sleplet._vars.SAMPLING_SCHEME,
        )
    )