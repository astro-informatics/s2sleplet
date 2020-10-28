from pathlib import Path

import numpy as np
import seaborn as sns

from pys2sleplet.plotting.create_plot import Plot

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
mask_path = file_location.parents[2] / "data" / "slepian" / "arbitrary" / "masks"
sns.set(context="paper")

L = 128


def plot_mask() -> None:
    """
    plots masks on the sphere without any harmonic transforms
    """
    name = f"south_america_L{L}"
    f = np.load(mask_path / f"{name}.npy").astype(np.complex128)
    Plot(f, L, L, f"mask_{name}").execute()


if __name__ == "__main__":
    plot_mask()
