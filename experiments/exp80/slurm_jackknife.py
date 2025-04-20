#!/usr/bin/env python3


import math

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    time: str = "00:30",  # HH:MM
    mem_per_gpu: str = "10G",
    n_jacks: int = 1000,
    n_nodes: int = 5,
    qos: str = "debug",
):
    n_splits = n_nodes * 4
    split_size = math.ceil(n_jacks / n_splits)

    cmds = []
    for ii in range(n_splits):
        start, end = ii * split_size, (ii + 1) * split_size
        end = min(n_jacks, end)
        base_cmd = """../exp80/do_jackknife.py {seed}
        --samples-fname shape_samples.npz
        --tag {tag} --n-jacks {n_jacks} --start {start} --end {end}
        """
        base_cmd = " ".join(base_cmd.split())
        cmd = base_cmd.format(seed=seed, tag=tag, n_jacks=n_jacks, start=start, end=end)
        cmds.append(cmd)

    run_multi_gpu_job(
        cmds,
        jobname=f"{tag}_jack",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=n_nodes,
    )


if __name__ == "__main__":
    typer.run(main)
