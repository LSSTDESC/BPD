#!/usr/bin/env python3

import typer

from bpd import HOME_DIR
from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    n_repeats: int = 10,
    n_splits: int = 500,
    n_samples: int = 1000,
    time: str = "03:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    script_path = HOME_DIR / "experiments" / "exp72" / "get_error_on_error_subset.py"
    assert script_path.exists()

    base_cmd = """{script_path} {seed}
    --samples-plus-fpath {samples_plus_fpath}
    --samples-minus-fpath {samples_minus_fpath}
    --tag {tag} --n-repeats {n_repeats} --n-splits {n_splits} --n-samples {n_samples}
    """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        script_path=script_path,
        seed=seed,
        tag=tag,
        samples_plus_fpath=samples_plus_fpath,
        samples_minus_fpath=samples_minus_fpath,
        n_repeats=n_repeats,
        n_splits=n_splits,
        n_samples=n_samples,
    )
    run_single_gpu_job(
        cmd,
        jobname=f"{tag}_{seed}_err_on_err_sub",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
    )


if __name__ == "__main__":
    typer.run(main)
