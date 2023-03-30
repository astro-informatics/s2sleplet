"""
creates a mesh object
"""
from dataclasses import KW_ONLY

from pydantic.dataclasses import dataclass

import sleplet._mesh_methods
import sleplet._plotly_methods
import sleplet._validation


@dataclass(config=sleplet._validation.Validation)
class Mesh:
    name: str
    _: KW_ONLY
    number_basis_functions: int | None = None
    zoom: bool = False

    def __post_init_post_parse__(self) -> None:
        mesh_config = sleplet._mesh_methods.extract_mesh_config(self.name)
        self.camera_view = sleplet._plotly_methods.create_camera(
            mesh_config["CAMERA_X"],
            mesh_config["CAMERA_Y"],
            mesh_config["CAMERA_Z"],
            mesh_config["REGION_ZOOM"] if self.zoom else mesh_config["DEFAULT_ZOOM"],
            x_center=mesh_config["CENTER_X"] if self.zoom else 0,
            y_center=mesh_config["CENTER_Y"] if self.zoom else 0,
            z_center=mesh_config["CENTER_Z"] if self.zoom else 0,
        )
        self.colourbar_pos = (
            mesh_config["REGION_COLOURBAR_POS"]
            if self.zoom
            else mesh_config["DEFAULT_COLOURBAR_POS"]
        )
        self.vertices, self.faces = sleplet._mesh_methods.read_mesh(mesh_config)
        self.region = sleplet._mesh_methods.create_mesh_region(
            mesh_config, self.vertices
        )
        (
            self.mesh_eigenvalues,
            self.basis_functions,
            self.number_basis_functions,
        ) = sleplet._mesh_methods.mesh_eigendecomposition(
            self.name,
            self.vertices,
            self.faces,
            number_basis_functions=self.number_basis_functions,
        )
