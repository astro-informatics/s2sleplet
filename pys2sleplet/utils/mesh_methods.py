import glob
from pathlib import Path

import numpy as np
from box import Box
from igl import adjacency_matrix, massmatrix, read_triangle_mesh
from numpy import linalg as LA

from pys2sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"
MESHES: set[str] = {
    Path(x.removesuffix(".toml")).stem
    for x in glob.glob(str(_meshes_path / "regions" / "*.toml"))
}


def _read_toml(mesh_name: str) -> Box:
    """
    reads in the given mesh region settings file
    """
    return Box.from_toml(filename=_meshes_path / "regions" / f"{mesh_name}.toml")


def read_mesh(mesh_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    reads in the given mesh
    """
    data = _read_toml(mesh_name)
    vertices, faces = read_triangle_mesh(str(_meshes_path / "polygons" / data.FILENAME))
    return vertices, faces


def create_mesh_region(mesh_name: str, vertices: np.ndarray) -> np.ndarray:
    """
    creates the boolean region for the given mesh
    """
    data = _read_toml(mesh_name)
    return (
        (vertices[:, 0] > data.XMIN)
        & (vertices[:, 0] < data.XMAX)
        & (vertices[:, 1] > data.YMIN)
        & (vertices[:, 1] < data.YMAX)
        & (vertices[:, 2] > data.ZMIN)
        & (vertices[:, 2] < data.ZMAX)
    )


def _graph_laplacian(faces: np.ndarray) -> np.ndarray:
    """
    computes the graph laplacian L = D - A
    where D is the degree matrix and A is the adjacency matrix
    """
    a_matrix = adjacency_matrix(faces).toarray()
    degrees = a_matrix.sum(axis=1)
    d_matrix = np.diagflat(degrees)
    return a_matrix - d_matrix


def mesh_eigendecomposition(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    computes the eigendecomposition of the mesh represented as a graph
    """
    logger.info(f"finding {vertices.shape[0]} basis functions of mesh")
    laplacian = _graph_laplacian(faces)
    return clean_evals_and_evecs(LA.eigh(laplacian))


def integrate_whole_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    function: np.ndarray,
) -> float:
    """
    computes the integral of a function defined on the vertices of the mesh
    """
    mass = massmatrix(vertices, faces)
    return mass.dot(function).sum()


def integrate_region_mesh(
    vertices: np.ndarray, faces: np.ndarray, function: np.ndarray, mask: np.ndarray
) -> float:
    """
    computes the integral of a region of a function defines of a mesh vertices
    """
    mass = massmatrix(vertices, faces)
    return mass.dot(function * mask).sum()


def clean_evals_and_evecs(
    eigendecomposition: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    need eigenvalues and eigenvectors to be in a certain format
    """
    # access values
    eigenvalues, eigenvectors = eigendecomposition

    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].conj().T

    # ensure first element of each eigenvector is positive
    eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]
    return eigenvalues, eigenvectors
