#!/usr/bin/env python3
import math

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    n_boots: int = 500,
    n_nodes: int = 5,
    time: str = "00:29",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "debug",
):
    n_splits = n_nodes * 4
    split_size = math.ceil(n_boots / n_splits)

    cmds = []
    for ii in range(n_splits):
        base_cmd = """../exp80/do_bootstrap.py {new_seed}
        --samples-fname shape_samples.npz
        --tag {tag} --n-boots {split_size}
        """
        base_cmd = " ".join(base_cmd.split())
        new_seed = f"{seed}{ii}"
        cmd = base_cmd.format(new_seed=new_seed, tag=tag, split_size=split_size)
        cmds.append(cmd)

    run_multi_gpu_job(
        cmds,
        jobname=f"{tag}_boot",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=n_nodes,
    )


if __name__ == "__main__":
    typer.run(main)
