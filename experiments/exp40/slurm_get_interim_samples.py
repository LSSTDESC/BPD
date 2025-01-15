#!/usr/bin/env python3
import subprocess

import typer

from bpd.slurm import setup_sbatch_job_gpu


def main(
    seed: int,
    time: str = "00:29",  # HH:MM
    mem_per_gpu: str = "25G",
    qos: str = "debug",  # debug (< 30 min), regular
):
    jobname = f"exp40_{seed}"

    jobfile = setup_sbatch_job_gpu(jobname, time, nodes=1, n_tasks_per_node=4, qos=qos)

    run_path = "/global/u2/i/imendoza/BPD/experiments/exp40/get_interim_samples.py"
    template_cmd = "python {run_path} {{seed}} {jobname}"

    base_cmd = template_cmd.format(run_path=run_path, jobname=jobname)

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
