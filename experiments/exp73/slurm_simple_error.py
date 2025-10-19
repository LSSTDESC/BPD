#!/usr/bin/env python3

import typer

from bpd import HOME_DIR
from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    time: str = "05:30",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    script_path = HOME_DIR / "experiments" / "exp73" / "simple_error.py"
    base_cmd = """python {script}
                {seed} --tag {tag}
                --samples-plus-fpath {samples_plus_fpath}
                --samples-minus-fpath {samples_minus_fpath}
                """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        script=script_path,
        seed=seed,
        tag=tag,
        samples_plus_fpath=samples_plus_fpath,
        samples_minus_fpath=samples_minus_fpath,
    )

    run_single_gpu_job(
        cmd,
        jobname=f"exp{tag}_simple_error",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
    )


if __name__ == "__main__":
    typer.run(main)
