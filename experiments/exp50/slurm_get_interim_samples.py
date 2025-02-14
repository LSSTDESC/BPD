#!/usr/bin/env python3

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    mode: str = "",
    time: str = "00:25",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "debug",
    nodes: int = 1,
    n_tasks_per_node: int = 4,
    g1: float = 0.02,
    g2: float = 0.0,
    n_gals: int = 2500,
    n_samples_per_gal: int = 300,
    mean_logflux: float = 4.0,
    sigma_logflux: float = 0.3,
    mean_loghlr: float = -0.1,
    sigma_loghlr: float = 0.05,
    shape_noise: float = 0.01,
    sigma_e_int: float = 0.05,
):
    mode_txt = f"_{mode}" if mode else ""
    jobname = f"{tag}{mode_txt}"

    base_cmd = """python /global/u2/i/imendoza/BPD/experiments/exp40/get_interim_samples.py
               {{seed}} {tag}
               --mode {mode}
               --g1 {g1} --g2 {g2}
               --n-gals {n_gals}
               --n-samples-per-gal {n_samples_per_gal}
               --mean-logflux {mean_logflux}
               --sigma-logflux {sigma_logflux}
               --mean-loghlr {mean_loghlr}
               --sigma-loghlr {sigma_loghlr}
               --shape-noise {shape_noise} --sigma-e-int {sigma_e_int}
               """
    base_cmd = " ".join(base_cmd.split())
    base_cmd = base_cmd.format(
        tag=tag,
        mode=mode,
        g1=g1,
        g2=g2,
        n_gals=n_gals,
        n_samples_per_gal=n_samples_per_gal,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
        shape_noise=shape_noise,
        sigma_e_int=sigma_e_int,
    )

    run_multi_gpu_job(
        base_cmd,
        jobname=jobname,
        base_seed=seed,
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=nodes,
        n_tasks_per_node=n_tasks_per_node,
    )


if __name__ == "__main__":
    typer.run(main)
