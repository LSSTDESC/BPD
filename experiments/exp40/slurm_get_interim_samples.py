#!/usr/bin/env python3

import typer

from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    mode: str = "",
    time: str = "00:25",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
    g1: float = 0.02,
    g2: float = 0.0,
    n_gals: int = 2500,
    n_samples_per_gal: int = 150,
    mean_logflux: float = 2.6,
    sigma_logflux: float = 0.4,
    shape_noise: float = 0.1,
    sigma_e_int: float = 0.15,
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
        nodes=1,
        n_tasks_per_node=4,
    )


if __name__ == "__main__":
    typer.run(main)
