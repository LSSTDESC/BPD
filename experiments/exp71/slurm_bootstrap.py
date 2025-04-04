#!/usr/bin/env python3
import math

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    time: str = "00:30",  # HH:MM
    samples_plus_fname: str = typer.Option(),
    samples_minus_fname: str = typer.Option(),
    mem_per_gpu: str = "10G",
    n_boots: int = 25,
    n_nodes: int = 5,
    qos: str = "debug",
):
    n_splits = n_nodes * 4
    split_size = math.ceil(n_boots / n_splits)

    cmds = []
    for ii in range(n_splits):
        base_cmd = """./simple_bootstrap.py {new_seed}
        --samples-plus-fname {samples_plus_fname}
        --samples-minus-fname {samples_minus_fname}
        --tag {tag} --n-boots {split_size}
        """
        base_cmd = " ".join(base_cmd.split())
        new_seed = f"{seed}{ii}"
        cmd = base_cmd.format(
            new_seed=new_seed,
            tag=tag,
            samples_plus_fname=samples_plus_fname,
            samples_minus_fname=samples_minus_fname,
            n_boots=split_size,
        )
        cmds.append(cmd)

    run_multi_gpu_job(
        cmds,
        jobname=f"{tag}_boots",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=n_nodes,
    )


if __name__ == "__main__":
    typer.run(main)
