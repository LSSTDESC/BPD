#!/usr/bin/env python3
import subprocess

import typer

from bpd import DATA_DIR
from bpd.slurm import setup_sbatch_job_gpu


def main(
    seed: int,
    jobname: str,
    time: str = "00:20",  # HH:MM
    mem_per_gpu: str = "20G",
    qos: str = "debug",  # debug (< 30 min), regular
    n_vec: int = 50,
    n_exps: int = 250,  # per gpu
    n_samples_per_gal: int = 1000,
    trim: int = 10,
    n_gals: int = 1000,
    n_samples_shear: int = 3000,
    g1: float = 0.02,
    g2: float = 0.0,
    add_extra: bool = False,
):
    tagpath = DATA_DIR / "cache_chains" / jobname
    if not add_extra:
        assert not tagpath.exists()

    jobfile = setup_sbatch_job_gpu(jobname, time, nodes=1, n_tasks_per_node=4, qos=qos)

    template_cmd = "python /global/u2/i/imendoza/BPD/scripts/toy_shear_vectorized.py {{seed}} {jobname} --n-gals {n_gals} --n-samples-shear {n_samples_shear} --n-vec {n_vec} --n-exps {n_exps} --n-samples-per-gal {n_samples_per_gal} --trim {trim} --g1 {g1} --g2 {g2}"

    base_cmd = template_cmd.format(
        n_gals=n_gals,
        n_samples_shear=n_samples_shear,
        n_vec=n_vec,
        n_exps=n_exps,
        jobname=jobname,
        n_samples_per_gal=n_samples_per_gal,
        trim=trim,
        g1=g1,
        g2=g2,
    )

    # append to jobfile the  commands.
    with open(jobfile, "a", encoding="utf-8") as f:
        f.write("\n")

    for ii in range(4):
        cmd_seed = int(f"{seed}{ii}")
        cmd = base_cmd.format(seed=cmd_seed)
        srun_cmd = f"srun --exact -u -n 1 -c 1 --gpus-per-task 1 --mem-per-gpu={mem_per_gpu} {cmd}  &\n"

        with open(jobfile, "a", encoding="utf-8") as f:
            f.write(srun_cmd)

    with open(jobfile, "a", encoding="utf-8") as f:
        f.write("\nwait")

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True, check=False)


if __name__ == "__main__":
    typer.run(main)
