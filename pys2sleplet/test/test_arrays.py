import numpy as np
from numpy.testing import assert_array_equal

from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix


def test_fill_matrix_using_hermitian_relation() -> None:
    """
    test that Hermitian symmetry is applied to matrix
    """
    matrix_in = np.array(
        [[1 + 1j, 0, 0], [4 + 4j, 5 + 5j, 0], [7 + 7j, 8 + 8j, 9 + 9j]]
    )
    matrix_out = np.array(
        [[1 + 1j, 4 - 4j, 7 - 7j], [4 + 4j, 5 + 5j, 8 - 8j], [7 + 7j, 8 + 8j, 9 + 9j]]
    )
    fill_upper_triangle_of_hermitian_matrix(matrix_in)
    assert_array_equal(matrix_in, matrix_out)