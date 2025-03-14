#!/usr/bin/env python3


from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str,
    time: str = "00:29",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "debug",
):
    base_cmd = "python /global/u2/i/imendoza/BPD/experiments/exp22/get_samples_full.py {seed} {tag}"
    cmds = []

    for ii in range(4):
        cmds.append(base_cmd.format(seed=f"{seed}{ii}", tag=tag))

    run_multi_gpu_job(
        cmds,
        jobname="full_samples_exp22",
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=1,
        n_tasks_per_node=4,
    )
