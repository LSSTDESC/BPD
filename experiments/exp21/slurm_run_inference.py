#!/usr/bin/env python3


from bpd.slurm import run_single_gpu_job


def main(
    seed: int,
    tag: str,
    time: str = "02:00",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "regular",
    mean_logflux: float = 3.0,
    sigma_logflux: float = 0.4,
    mean_loghlr: float = 0.0,
    sigma_loghlr: float = 0.05,
    min_logflux: float = 2.4,
):
    base_cmd = """python /global/u2/i/imendoza/BPD/experiments/exp21/run_inference_galaxy_images.py
                {seed} {tag}
                --mean-logflux {mean_logflux}
                --sigma-logflux {sigma_logflux}
                --mean-loghlr {mean_loghlr}
                --sigma-loghlr {sigma_loghlr}
                --min-logflux {min_logflux}
                --mode 'long'
                """
    base_cmd = " ".join(base_cmd.split())
    cmd = base_cmd.format(
        seed=seed,
        tag=tag,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
        min_logflux=min_logflux,
    )

    run_single_gpu_job(
        cmd,
        jobname=tag,
        base_seed=seed,
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=1,
        n_tasks_per_node=1,
    )
