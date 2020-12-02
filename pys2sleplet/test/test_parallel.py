from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_LARGE, L_SMALL, NCPU, ORDER, THETA_MAX
from pys2sleplet.utils.parallel_methods import split_L_into_chunks


def test_slepian_polar_cap_serial_equal_to_parallel() -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian polar cap give the same result
    """
    serial = SlepianPolarCap(L_SMALL, THETA_MAX, order=ORDER, ncpu=1)
    parallel = SlepianPolarCap(L_SMALL, THETA_MAX, order=ORDER)
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)


def test_split_L_into_chunks() -> None:
    """
    ensure vector L split into appropriate number of chunks
    """
    chunks = split_L_into_chunks(L_SMALL, NCPU)
    assert_equal(len(chunks), NCPU)
    chunk_length = L_SMALL // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=0)


def test_split_L_into_chunks_Lmin_Lmax() -> None:
    """
    ensure vector L split into appropriate number of chunks with L_min
    """
    L_max = L_LARGE + 1  # want to test odd number
    chunks = split_L_into_chunks(L_max, NCPU, L_min=L_SMALL)
    assert_equal(len(chunks), NCPU)
    chunk_length = (L_max - L_SMALL) // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=1)
