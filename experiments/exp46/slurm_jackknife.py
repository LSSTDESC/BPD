#!/usr/bin/env python3


import typer

from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    time: str = "04:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    base_cmd = """../exp40/simple_jackknife.py {seed}
	--samples-plus-fname interim_samples_{seed}_plus.npz
	--samples-minus-fname interim_samples_{seed}_minus.npz
	--tag {tag} --overwrite --n-jacks 100
    """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        seed=seed,
        tag=tag,
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
