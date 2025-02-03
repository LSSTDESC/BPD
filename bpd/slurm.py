"""
Submit sbatch job with default settings. To be used in other scripts specifying `cmd`.
"""

import secrets
import subprocess
from pathlib import Path

JOB_DIR = Path(__file__).parent.parent.joinpath("jobs_out")
assert JOB_DIR.exists(), f"Need to create '{JOB_DIR}' dir"
JOB_SEED = secrets.randbelow(1_000_000_000)


def setup_sbatch_job_gpu(
    jobname: str,
    time: str = "00:10",
    nodes: int = 1,
    n_tasks_per_node: int = 1,
    qos: str = "regular",
) -> Path:
    """Returns path to sbatch ready job file, commands need to be appended."""

    # time formating
    assert len(time) == 5 and time.count(":") == 1

    # prepare files and directories
    jobfile_name = f"{jobname}_{JOB_SEED}.sbatch"
    jobfile = JOB_DIR.joinpath(jobfile_name)

    with open(jobfile, "w", encoding="utf-8") as f:
        f.writelines(
            "#!/bin/bash\n"
            f"#SBATCH --job-name={jobname}\n"
            f"#SBATCH --account=m1727\n"
            f"#SBATCH --qos={qos}\n"
            f"#SBATCH --mail-type=begin,end,fail\n"
            f"#SBATCH --mail-user=imendoza@umich.edu\n"
            f"#SBATCH -C gpu\n"
            f"#SBATCH --output={JOB_DIR}/{jobname}_%j.out\n"
            f"#SBATCH --time={time}:00\n"
            f"#SBATCH --nodes={nodes}\n"
            f"#SBATCH --ntasks-per-node={n_tasks_per_node}\n"
        )

    return jobfile


def run_multi_gpu_job(
    base_cmd: str,
    *,
    jobname: str,
    base_seed: int,
    time: str = "00:25",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
    nodes: int = 1,
    n_tasks_per_node: int = 4,
):
    jobfile = setup_sbatch_job_gpu(
        jobname, time=time, nodes=nodes, n_tasks_per_node=n_tasks_per_node, qos=qos
    )
    assert "{seed}" in base_cmd

    # append to jobfile the  commands.
    with open(jobfile, "a", encoding="utf-8") as f:
        f.write("\n")

    for ii in range(n_tasks_per_node):
        cmd_seed = int(f"{base_seed}{ii}")
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
