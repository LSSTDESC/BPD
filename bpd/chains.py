import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from numpyro.infer.mcmc import MCMC, MCMCKernel


def _collect_samples(samples: dict, all_samples: dict, axis: int = 0):
    """Collect samples from `samples` into `all_samples`."""
    for k, v1 in samples.items():
        if k not in all_samples:
            all_samples[k] = v1
        else:
            v = all_samples[k]
            for ii, s in enumerate(v.shape):
                if ii != axis and ii - v.ndim != axis:
                    assert s == v1.shape[ii]
            all_samples[k] = jnp.concatenate([v, v1], axis=axis)
    return all_samples


def _check_dict_shapes(samples: dict, shape: tuple):
    """Check that all samples have shape (n_samples, ...)"""
    for _, v in samples.items():
        assert v.ndim == len(shape)
        for ii, s in enumerate(shape):
            if s != -1:
                assert v.shape[ii] == s


def run_chains(
    data: jax.Array | np.ndarray,
    kernel: MCMCKernel,
    n_vec: int = 1,
    n_warmup: int = 100,
    n_samples: int = 1000,
    seed: int = 0,
):
    """Run chains on subsets of data as iid samples and collect samples.

    This function is particularly useful when we want to run on multiple noise
    realizations of the same galaxy image. Or just generally on pre-processed data.

    Args:
        data: (n, ...) array of data
        kernel: MCMC kernel
        seed: random seed
        n_vec: number of samples to vectorize over
        n_warmup: number of warmup steps
        n_samples: number of total samples (produced after warmup)
    """
    n = len(data)
    all_samples = {}
    rng_key = PRNGKey(seed)
    for ii in range(0, n, n_vec):
        data_ii = data[ii : ii + n_vec] if n_vec > 1 else data[ii]
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples)
        mcmc.run(rng_key, data=data_ii)  # reuse key is OK, different data

        samples = mcmc.get_samples()
        samples = {k: v.reshape(n_samples, -1, n_vec) for k, v in samples.items()}
        all_samples = _collect_samples(samples, all_samples, axis=-1)

    all_samples = {k: v.transpose((2, 0, 1)) for k, v in all_samples.items()}
    _check_dict_shapes(all_samples, (n, n_samples, -1))
    return all_samples


def run_pmap_not_vectorized(
    data: jax.Array | np.ndarray,
    kernel: MCMCKernel,
    n_warmup: int = 100,
    n_samples: int = 1000,
    seed: int = 0,
):
    """Run chains in parallel over gpus over each image in `data`, no vectorization."""
    n = len(data)
    all_samples = {}
    rng_key = PRNGKey(seed)

    # get number of gpus available
    n_gpus = jax.local_device_count()

    # split data into chunks
    chunk_size = n // n_gpus
    data_chunks = [data[i : i + chunk_size] for i in range(0, n, chunk_size)]

    def run_chain(data_chunk):
        n_images = len(data_chunk)
        samples = {}
        for ii in range(n_images):
            data = data_chunk[ii]
            mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples)
            mcmc.run(rng_key, data=data)
            samples_ii = mcmc.get_samples()
            samples_ii = {k: v.reshape(1, n_samples, -1) for k, v in samples_ii.items()}
            samples = _collect_samples(samples_ii, samples, axis=0)
        return samples

    samples_chunks = jax.pmap(run_chain)(data_chunks)

    for samples in samples_chunks:
        all_samples = _collect_samples(samples, all_samples, axis=0)

    return all_samples
