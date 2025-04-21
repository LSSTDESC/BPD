#!/usr/bin/env python3

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    time: str = typer.Option(),
    mode: str = typer.Option(),
    qos: str = typer.Option(),
    nodes: int = typer.Option(),
    n_tasks_per_node: int = 4,
    g1: float = typer.Option(),
    g2: float = 0.0,
    n_gals: int = typer.Option(),
    n_samples_per_gal: int = 300,
    mem_per_gpu: str = "10G",
):
    assert mode in ("plus", "minus", "")
    mode_txt = f"_{mode}" if mode else ""
    jobname = f"{tag}{mode_txt}"

    base_cmd = """python /global/u2/i/imendoza/BPD/experiments/exp70/get_interim_samples.py
               {{seed}} --tag {tag}
               --mode {mode}
               --g1 {g1} --g2 {g2}
               --n-gals {n_gals}
               --n-samples-per-gal {n_samples_per_gal}
               """
    base_cmd = " ".join(base_cmd.split())
    base_cmd = base_cmd.format(
        tag=tag,
        mode=mode,
        g1=g1,
        g2=g2,
        n_gals=n_gals,
        n_samples_per_gal=n_samples_per_gal,
    )

    cmds = []
    for ii in range(nodes):
        for jj in range(n_tasks_per_node):
            _seed = f"{seed}{ii}{jj}"
            cmd = base_cmd.format(seed=_seed)
            cmds.append(cmd)

    run_multi_gpu_job(
        cmds,
        jobname=jobname,
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=nodes,
        n_tasks_per_node=n_tasks_per_node,
    )


if __name__ == "__main__":
    typer.run(main)
