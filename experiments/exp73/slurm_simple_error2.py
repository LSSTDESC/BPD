#!/usr/bin/env python3

import typer

from bpd import HOME_DIR
from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str = typer.Option(),
    posterior_fpath: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    n_splits: int = 500,
    n_samples: int = 1000,
    initial_step_size: float = 0.01,
    time: str = "04:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
):
    script_path = HOME_DIR / "experiments" / "exp73" / "simple_error2.py"
    base_cmd = """python {script}
                {seed} --tag {tag}
                --posterior-fpath {posterior_fpath}
                --samples-plus-fpath {samples_plus_fpath}
                --samples-minus-fpath {samples_minus_fpath}
                --n-splits {n_splits}
                --n-samples {n_samples}
                --initial-step-size {initial_step_size}
                --start {{start}}
                --end {{end}}
                """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        script=script_path,
        seed=seed,
        tag=tag,
        posterior_fpath=posterior_fpath,
        samples_plus_fpath=samples_plus_fpath,
        samples_minus_fpath=samples_minus_fpath,
        n_splits=n_splits,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    cmds = []
    assert n_splits % 125 == 0, "n_splits must be divisible by 125"
    assert n_splits // 125 == 4
    for ii in range(0, 4):
        start = ii * 125
        end = (ii + 1) * 125
        cmd_ = cmd.format(start=start, end=end)
        cmds.append(cmd_)

    run_multi_gpu_job(
        cmds,
        jobname="exp73_simple_error2",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=1,
        n_tasks_per_node=4,
    )


if __name__ == "__main__":
    typer.run(main)
