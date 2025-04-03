#!/usr/bin/env python3
import math

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    n_exps: int = 500,
    n_nodes: int = 5,
    shape_noise: float = 1e-2,
    sigma_e_int: float = 5e-2,
    n_gals: int = 10_000,
    time: str = "00:15",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "debug",
):
    n_splits = n_nodes * 4
    split_size = math.ceil(n_exps / n_splits)

    cmds = []
    for ii in range(n_splits):
        base_cmd = """../exp80/do_mc_shear.py {new_seed}
        --tag {tag} --n-exps {split_size} --shape-noise {shape_noise} --sigma-e-int {sigma_e_int}
        --n-gals {n_gals}
        """
        base_cmd = " ".join(base_cmd.split())
        new_seed = f"{seed}{ii}"
        cmd = base_cmd.format(
            new_seed=new_seed,
            tag=tag,
            split_size=split_size,
            shape_noise=shape_noise,
            sigma_e_int=sigma_e_int,
            n_gals=n_gals,
        )
        cmds.append(cmd)

    run_multi_gpu_job(
        cmds,
        jobname=f"{tag}_mc",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=n_nodes,
    )


if __name__ == "__main__":
    typer.run(main)
