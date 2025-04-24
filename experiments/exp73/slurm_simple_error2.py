#!/usr/bin/env python3

import typer

from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    posterior_fpath: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    n_splits: int = 500,
    n_samples: int = 1000,
    initial_step_size: float = 0.01,
    time: str = "03:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    base_cmd = """python /global/u2/i/imendoza/BPD/experiments/exp73/simple_error2.py
                {seed} --tag {tag}
                --posterior-fpath {posterior_fpath}
                --samples-plus-fpath {samples_plus_fpath}
                --samples-minus-fpath {samples_minus_fpath}
                --n-splits {n_splits}
                --n-samples {n_samples}
                --initial-step-size {initial_step_size}
                """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        seed=seed,
        tag=tag,
        posterior_fpath=posterior_fpath,
        samples_plus_fpath=samples_plus_fpath,
        samples_minus_fpath=samples_minus_fpath,
        n_splits=n_splits,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    run_single_gpu_job(
        cmd,
        jobname=tag + "_simple_error2",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
    )


if __name__ == "__main__":
    typer.run(main)
