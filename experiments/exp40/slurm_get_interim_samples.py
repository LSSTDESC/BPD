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
    mode: str = "",
):
    mode_txt = f"_{mode}" if mode else ""

    jobfile = setup_sbatch_job_gpu(
        f"{tag}{mode_txt}", time=time, nodes=1, n_tasks_per_node=4, qos=qos
    )

    run_path = "/global/u2/i/imendoza/BPD/experiments/exp40/get_interim_samples.py"
    template_cmd = "python {run_path} {{seed}} {tag} --mode {mode} --g1 {g1} --g2 {g2}"

    base_cmd = template_cmd.format(run_path=run_path, tag=tag, mode=mode, g1=g1, g2=g2)

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
