import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from numpyro.infer.mcmc import MCMC, MCMCKernel


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
        data_ii = data[ii : ii + n_vec]
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples)
        mcmc.run(rng_key, data=data_ii)  # reuse key is OK, different data
        samples = mcmc.get_samples()
        for k in samples:
            v2 = samples.get(k)
            assert v2.shape == (n_samples, n_vec)
            if k not in all_samples:
                all_samples[k] = v2
            else:
                v1 = all_samples[k]
                assert v1.shape[0] == n_samples
                assert v1.ndim == 2
                all_samples[k] = jnp.concatenate([v1, v2], axis=-1)
    all_samples = {k: v.T for k, v in all_samples.items()}

    for k, v in all_samples.items():
        assert v.shape == (n, n_samples)

    return all_samples


def run_pmap_not_vectorized(
    data: jax.Array | np.ndarray,
    kernel: MCMCKernel,
    n_warmup: int = 100,
    n_samples: int = 1000,
    seed: int = 0,
):
    """Run chains in parallel over gpus over each image in `data`, no vectorization."""
    # TODO: change so that we run each chunk in a for loop
    # TODO: consider refactoring logic of collecting samples into one dictionary in separate func.
    n = len(data)
    all_samples = {}
    rng_key = PRNGKey(seed)

    # get number of gpus available
    n_gpus = jax.local_device_count()

    # split data into chunks
    chunk_size = n // n_gpus
    data_chunks = [data[i : i + chunk_size] for i in range(0, n, chunk_size)]

    # run chains in parallel
    def run_chain(data_chunk):
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples)
        mcmc.run(rng_key, data=data_chunk)
        return mcmc.get_samples()

    samples_chunks = jax.pmap(run_chain)(data_chunks)

    # collect samples
    for samples in samples_chunks:
        for k, v in samples.items():
            if k not in all_samples:
                all_samples[k] = v
            else:
                all_samples[k] = jnp.concatenate([all_samples[k], v], axis=0)

    return all_samples
