"""
Submit sbatch job with default settings. To be used in other scripts specifying `cmd`.
"""

import secrets
from pathlib import Path

JOB_DIR = Path(__file__).parent.joinpath("jobs_out")
assert JOB_DIR.exists(), f"Need to create '{JOB_DIR}' dir"
JOB_SEED = secrets.randbelow(1_000_000_000)


def setup_sbatch_job_gpu(
    jobname: str,
    time: str = "00:10",
    nodes: int = 1,
    n_tasks_per_node: int = 1,
    qos: str = "debug",
) -> Path:
    """Returns path to sbatch ready job file, commands need to be appended."""

    # time formating
    assert len(time) == 5 and time.count(":") == 1

    # prepare files and directories
    jobfile_name = f"{jobname}_{JOB_SEED}.sbatch"
    job_dir = Path(JOB_DIR)
    jobfile = job_dir.joinpath(jobfile_name)

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
