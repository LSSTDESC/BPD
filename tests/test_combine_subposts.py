import numpy as np
from numpy.linalg import inv

from bpd.utils import combine_subposts_gaussian


def test_gauss_subposts_easy():
    n_samples = 1_000_000
    rng = np.random.default_rng(42)

    mu1 = np.array([1.0, 0.5])
    mu2 = np.array([2.0, 1.0])
    _cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    samples1 = rng.multivariate_normal(mean=mu1, cov=_cov, size=(n_samples,))
    samples2 = rng.multivariate_normal(mean=mu2, cov=_cov, size=(n_samples,))

    samples = np.stack([samples1, samples2])
    assert samples.shape == (2, n_samples, 2)

    mu, cov = combine_subposts_gaussian(samples)

    # not ok at .1% level
    assert not np.allclose(mu.reshape(-1), np.array([1.5, 0.75]), rtol=0.001)
    assert not np.allclose(cov, 0.5 * _cov, rtol=0.001)

    np.testing.assert_allclose(mu.reshape(-1), np.array([1.5, 0.75]), rtol=0.01)
    np.testing.assert_allclose(cov, 0.5 * _cov, rtol=0.01)

    # !!!
    np.testing.assert_allclose(samples.mean((0, 1)), np.array([1.5, 0.75]), rtol=0.001)


def test_gauss_subposts_few_samples():
    # what happens if we do not have enough samples
    n_samples = 1_000
    rng = np.random.default_rng(42)

    mu1 = np.array([1.0, 0.5])
    mu2 = np.array([2.0, 1.0])
    _cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    samples1 = rng.multivariate_normal(mean=mu1, cov=_cov, size=(n_samples,))
    samples2 = rng.multivariate_normal(mean=mu2, cov=_cov, size=(n_samples,))

    samples = np.stack([samples1, samples2])

    mu, cov = combine_subposts_gaussian(samples)

    # not ok at 1% level
    assert not np.allclose(mu.reshape(-1), np.array([1.5, 0.75]), rtol=0.01)
    assert not np.allclose(cov, 0.5 * _cov, rtol=0.01)

    # ok at 10% level
    np.testing.assert_allclose(mu.reshape(-1), np.array([1.5, 0.75]), rtol=0.1)
    np.testing.assert_allclose(cov, 0.5 * _cov, rtol=0.1)


def test_gauss_subposts():
    n_samples = 1_000_000
    rng = np.random.default_rng(42)

    mu1 = np.array([1.0, 0.5])
    mu2 = np.array([2.0, 1.0])
    cov1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov2 = np.array([[1.0, 0.9], [0.9, 1.0]])

    samples1 = rng.multivariate_normal(mean=mu1, cov=cov1, size=(n_samples,))
    samples2 = rng.multivariate_normal(mean=mu2, cov=cov2, size=(n_samples,))

    samples = np.stack([samples1, samples2])

    # function
    fmu1, fcov1 = combine_subposts_gaussian(samples)

    # manual
    fcov2 = inv(inv(cov1) + inv(cov2))
    fmu2 = fcov2.dot(
        inv(cov1).dot(mu1.reshape(-1, 1)) + inv(cov2).dot(mu2.reshape(-1, 1))
    )

    np.testing.assert_allclose(fmu1, fmu2, rtol=0.01)
    np.testing.assert_allclose(fcov1, fcov2, rtol=0.01)

    # not ok at .1% level
    assert not np.allclose(fmu1, fmu2, rtol=0.001)
    assert not np.allclose(fcov1, fcov2, rtol=0.001)
