#!/usr/bin/env python3
import subprocess

import click
from slurm_job import setup_sbatch_job_gpu

from bpd import DATA_DIR


@click.command()
@click.option("--seed", required=True, type=int)
@click.option("--jobname", required=True, type=str)
@click.option("--time", default="00:20", type=str, show_default=True, help="HH:MM")
@click.option("--mem-per-gpu", default="20G", type=str, show_default=True)
@click.option("--n-vec", default=50, type=int, show_default=True)
@click.option("--k", default=1000, type=int, show_default=True)
@click.option("--trim", default=10, type=int, show_default=True)
@click.option("--n-seeds-per-task", default=250, type=int, show_default=True)
@click.option("--n-gals", default=1000, type=int, show_default=True)
@click.option("--n-samples-shear", default=3000, type=int, show_default=True)
@click.option("--g1", type=float, default=0.02)
@click.option("--g2", type=float, default=0.0)
@click.option(
    "--add-extra",
    is_flag=True,
    show_default=True,
    default=False,
    help="Adding additional runs to previously existing experiment.",
)
@click.option("--qos", default="debug", type=str, show_default=True)  # debug, regular
def main(
    seed: int,
    jobname: str,
    time: str,
    mem_per_gpu: str,
    n_vec: int,
    k: int,
    trim: int,
    n_seeds_per_task: int,
    n_gals: int,
    n_samples_shear: int,
    g1: float,
    g2: float,
    add_extra: bool,
    qos: str,
):
    tagpath = DATA_DIR / "cache_chains" / jobname
    if not add_extra:
        assert not tagpath.exists()

    jobfile = setup_sbatch_job_gpu(jobname, time, nodes=1, n_tasks_per_node=4, qos=qos)

    template_cmd = "python /global/u2/i/imendoza/BPD/scripts/vect_toy_shear_gpu.py --n-samples-gals {n_gals} --n-samples-shear {n_samples_shear} --n-vec {n_vec} --seed {{seed}}  --n-seeds {n_seeds_per_task} --tag {jobname} --k {k} --trim {trim} --g1 {g1} --g2 {g2}"

    base_cmd = template_cmd.format(
        n_gals=n_gals,
        n_samples_shear=n_samples_shear,
        n_vec=n_vec,
        n_seeds_per_task=n_seeds_per_task,
        jobname=jobname,
        k=k,
        trim=trim,
        g1=g1,
        g2=g2,
    )

    # append to jobfile the  commands.
    with open(jobfile, "a") as f:
        f.write("\n")

    for ii in range(4):
        cmd_seed = int(f"{seed}{ii}")
        cmd = base_cmd.format(seed=cmd_seed)
        srun_cmd = f"srun --exact -u -n 1 -c 1 --gpus-per-task 1 --mem-per-gpu={mem_per_gpu} {cmd}  &\n"

        with open(jobfile, "a") as f:
            f.write(srun_cmd)

    with open(jobfile, "a") as f:
        f.write("\nwait")

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True, check=False)


if __name__ == "__main__":
    main()
