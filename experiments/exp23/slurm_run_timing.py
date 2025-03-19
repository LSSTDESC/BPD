#!/usr/bin/env python3


import typer

from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str,
    time: str = "02:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    cmd = f"python /global/u2/i/imendoza/BPD/experiments/exp23/get_samples_timing.py {seed} {tag}"
    run_single_gpu_job(
        cmd,
        jobname=tag,
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
    )


if __name__ == "__main__":
    typer.run(main)
