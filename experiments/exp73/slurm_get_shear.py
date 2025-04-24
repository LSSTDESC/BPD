#!/usr/bin/env python3

import typer

from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_fpath: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_samples: int = 3000,
    extra_tag: str = "",
    time: str = "03:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    base_cmd = """python /global/u2/i/imendoza/BPD/experiments/exp73/get_shear.py
                {seed} --tag {tag}
                --samples-fpath {samples_fpath}
                --initial-step-size {initial_step_size}
                --n-samples {n_samples}
                --extra-tag {extra_tag}
                --overwrite
                """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        seed=seed,
        tag=tag,
        samples_fpath=samples_fpath,
        initial_step_size=initial_step_size,
        n_samples=n_samples,
        extra_tag=extra_tag,
    )

    run_single_gpu_job(
        cmd,
        jobname=tag,
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
    )


if __name__ == "__main__":
    typer.run(main)
