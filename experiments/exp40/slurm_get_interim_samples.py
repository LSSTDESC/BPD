#!/usr/bin/env python3
import subprocess

import typer

from bpd.slurm import setup_sbatch_job_gpu


def main(
    seed: int,
    tag: str = typer.Option(),
    time: str = "00:25",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "debug",  # debug (< 30 min), regular
    g1: float = 0.02,
    g2: float = 0.0,
    n_gals: int = 2500,
    n_samples_per_gal: int = 150,
    mean_logflux: float = 2.6,
    sigma_logflux: float = 0.4,
    shape_noise: float = 0.1,
    sigma_e_int: float = 0.15,
    mode: str = "",
):
    mode_txt = f"_{mode}" if mode else ""

    jobfile = setup_sbatch_job_gpu(
        f"{tag}{mode_txt}", time=time, nodes=1, n_tasks_per_node=4, qos=qos
    )

    run_path = "/global/u2/i/imendoza/BPD/experiments/exp40/get_interim_samples.py"
    template_cmd = "python {run_path} {{seed}} {tag} --mode {mode} --g1 {g1} --g2 {g2} --n-gals {n_gals} --n-samples-per-gal {n_samples_per_gal} --mean-logflux {mean_logflux} --sigma-logflux {sigma_logflux} --shape-noise {shape_noise} --sigma-e-int {sigma_e_int}"

    base_cmd = template_cmd.format(
        run_path=run_path,
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

    # append to jobfile the  commands.
    with open(jobfile, "a", encoding="utf-8") as f:
        f.write("\n")

    for ii in range(4):
        cmd_seed = int(f"{seed}{ii}")
        cmd = base_cmd.format(seed=cmd_seed)
        srun_cmd = (
            f"srun --exact -u -n 1 -c 1 --gpus-per-task 1 "
            f"--mem-per-gpu={mem_per_gpu} {cmd}  &\n"
        )

        with open(jobfile, "a", encoding="utf-8") as f:
            f.write(srun_cmd)

    with open(jobfile, "a", encoding="utf-8") as f:
        f.write("\nwait")

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True, check=False)


if __name__ == "__main__":
    typer.run(main)
