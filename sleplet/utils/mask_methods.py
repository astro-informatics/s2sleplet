from pathlib import Path

import numpy as np
import pyssht as ssht
from box import Box

from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.harmonic_methods import mesh_forward, mesh_inverse
from sleplet.utils.logger import logger
from sleplet.utils.region import Region
from sleplet.utils.vars import SAMPLING_SCHEME

_file_location = Path(__file__).resolve()
_mask_path = _file_location.parents[1] / "data" / "slepian" / "masks"


def create_mask_region(L: int, region: Region) -> np.ndarray:
    """
    creates a mask of a region of interested, the output will be based
    on the value of the provided L. The mask could be either:
    * polar cap - if theta_max provided
    * limited latitude longitude - if one of theta_min, theta_max,
                                   phi_min or phi_max is provided
    * arbitrary - just checks the shape of the input mask
    """
    thetas, phis = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)

    if region.region_type == "arbitrary":
        logger.info("loading and checking shape of provided mask")
        name = f"{region.mask_name}_L{L}.npy"
        mask = _load_mask(name)
        assert mask.shape == thetas.shape, (
            f"mask {name} has shape {mask.shape} which does not match "
            f"the provided L={L}, the shape should be {thetas.shape}"
        )

    elif region.region_type == "lim_lat_lon":
        logger.info("creating limited latitude longitude mask")
        mask = (
            (thetas >= region.theta_min)
            & (thetas <= region.theta_max)
            & (phis >= region.phi_min)
            & (phis <= region.phi_max)
        )

    elif region.region_type == "polar":
        logger.info("creating polar cap mask")
        mask = thetas <= region.theta_max
        if region.gap:
            logger.info("creating polar gap mask")
            mask |= thetas >= np.pi - region.theta_max
    return mask


def _load_mask(mask_name: str) -> np.ndarray:
    """
    attempts to read the mask from the config file
    """
    location = _mask_path / mask_name
    try:
        mask = np.load(location)
    except FileNotFoundError:
        logger.error(f"can not find the file: '{mask_name}'")
        raise
    return mask


def ensure_masked_flm_bandlimited(
    flm: np.ndarray, L: int, region: Region, reality: bool, spin: int
) -> np.ndarray:
    """
    ensures the coefficients is bandlimited for a given region
    """
    field = ssht.inverse(flm, L, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME)
    mask = create_mask_region(L, region)
    field = np.where(mask, field, 0)
    return ssht.forward(field, L, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME)


def create_default_region(settings: Box) -> Region:
    """
    creates default region from settings object
    """
    return Region(
        gap=settings.POLAR_GAP,
        mask_name=settings.SLEPIAN_MASK,
        phi_max=np.deg2rad(settings.PHI_MAX),
        phi_min=np.deg2rad(settings.PHI_MIN),
        theta_max=np.deg2rad(settings.THETA_MAX),
        theta_min=np.deg2rad(settings.THETA_MIN),
    )


def create_mesh_region(mesh_config: Box, vertices: np.ndarray) -> np.ndarray:
    """
    creates the boolean region for the given mesh
    """
    return (
        (vertices[:, 0] >= mesh_config.XMIN)
        & (vertices[:, 0] <= mesh_config.XMAX)
        & (vertices[:, 1] >= mesh_config.YMIN)
        & (vertices[:, 1] <= mesh_config.YMAX)
        & (vertices[:, 2] >= mesh_config.ZMIN)
        & (vertices[:, 2] <= mesh_config.ZMAX)
    )


def ensure_masked_bandlimit_mesh_signal(mesh: Mesh, u_i: np.ndarray) -> np.ndarray:
    """
    ensures that signal in pixel space is bandlimited
    """
    field = mesh_inverse(mesh, u_i)
    masked_field = np.where(mesh.region, field, 0)
    return mesh_forward(mesh, masked_field)


def convert_region_on_vertices_to_faces(mesh: Mesh) -> np.ndarray:
    """
    converts the region on vertices to faces
    """
    region_reshape = np.argwhere(mesh.region).reshape(-1)
    faces_in_region = np.isin(mesh.faces, region_reshape).all(axis=1)
    region_on_faces = np.zeros(mesh.faces.shape[0])
    region_on_faces[faces_in_region] = 1
    return region_on_faces