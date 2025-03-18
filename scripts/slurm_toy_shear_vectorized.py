#!/usr/bin/env python3


from bpd import DATA_DIR
from bpd.slurm import run_multi_gpu_job


def main(
    seed: int,
    tag: str,
    time: str = "00:20",  # HH:MM
    mem_per_gpu: str = "10G",
    qos: str = "debug",
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
    tagpath = DATA_DIR / "cache_chains" / tag
    if not add_extra:
        assert not tagpath.exists()

    base_cmd = """python /global/u2/i/imendoza/BPD/scripts/toy_shear_vectorized.py
                {{seed}} {tag}
                --n-gals {n_gals} --n-samples-shear {n_samples_shear}
                --n-vec {n_vec} --n-exps {n_exps}
                --n-samples-per-gal {n_samples_per_gal}
                --trim {trim}
                --g1 {g1} --g2 {g2}
                """
    base_cmd = " ".join(base_cmd.split())
    base_cmd = base_cmd.format(
        n_gals=n_gals,
        n_samples_shear=n_samples_shear,
        n_vec=n_vec,
        n_exps=n_exps,
        tag=tag,
        n_samples_per_gal=n_samples_per_gal,
        trim=trim,
        g1=g1,
        g2=g2,
    )

    cmds = []
    for ii in range(1):
        for jj in range(4):
            _seed = f"{seed}{ii}{jj}"
            cmd = base_cmd.format(seed=_seed)
            cmds.append(cmd)

    run_multi_gpu_job(
        cmds,
        jobname=tag,
        time=time,
        mem_per_gpu=mem_per_gpu,
        qos=qos,
        nodes=1,
        n_tasks_per_node=4,
    )
