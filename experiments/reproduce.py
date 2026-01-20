
from generate import SbatchGenerator

run_group = "main-experiments"
data_root = ... # the root directory of OGBench 100M datasets

num_jobs_per_gpu = 1
array_limit = 100

agent_params = dict(
    FBRAC={
        "puzzle-3x3-play-sparse":       dict(alpha=0.03),
        "scene-play-sparse":            dict(alpha=100.0),
        "cube-double-play":             dict(alpha=0.1),
        "antmaze-large-navigate":       dict(alpha=0.1), 
        "humanoidmaze-medium-navigate": dict(alpha=30.0),

        "cube-triple-play":             dict(alpha=0.03),
        "cube-quadruple-play":          dict(alpha=1.0),
        "puzzle-4x4-play-sparse":       dict(alpha=0.3),
        "antmaze-giant-navigate":       dict(alpha=0.1), 
        "humanoidmaze-large-navigate":  dict(alpha=10.0),
    },
    IFQL={
        "puzzle-3x3-play-sparse":       dict(kappa=0.95),
        "scene-play-sparse":            dict(kappa=0.9),
        "cube-double-play":             dict(kappa=0.9),
        "antmaze-large-navigate":       dict(kappa=0.9), 
        "humanoidmaze-medium-navigate": dict(kappa=0.7),

        "cube-triple-play":             dict(kappa=0.95),
        "cube-quadruple-play":          dict(kappa=0.95),
        "puzzle-4x4-play-sparse":       dict(kappa=0.9),
        "antmaze-giant-navigate":       dict(kappa=0.8), 
        "humanoidmaze-large-navigate":  dict(kappa=0.8),
    },
    CGQL={
        "puzzle-3x3-play-sparse":       dict(mode="simple", inv_temp=10., guidance_coef=0.1),
        "scene-play-sparse":            dict(mode="simple", inv_temp=10., guidance_coef=0.1),
        "cube-double-play":             dict(mode="simple", inv_temp=10., guidance_coef=0.01),
        "antmaze-large-navigate":       dict(mode="simple", inv_temp=10., guidance_coef=0.1), 
        "humanoidmaze-medium-navigate": dict(mode="simple", inv_temp=10., guidance_coef=0.01),

        "cube-triple-play":             dict(mode="simple", inv_temp=10., guidance_coef=0.1),
        "cube-quadruple-play":          dict(mode="simple", inv_temp=10., guidance_coef=0.1),
        "puzzle-4x4-play-sparse":       dict(mode="simple", inv_temp=10., guidance_coef=0.1),
        "antmaze-giant-navigate":       dict(mode="simple", inv_temp=10., guidance_coef=0.1), 
        "humanoidmaze-large-navigate":  dict(mode="simple", inv_temp=10., guidance_coef=0.01),
    },
    CGQL_MSE={
        "puzzle-3x3-play-sparse":       dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),
        "scene-play-sparse":            dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),
        "cube-double-play":             dict(mode="mse", inv_temp=10., guidance_coef=0.01, noisy_coef=0.001),
        "antmaze-large-navigate":       dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1), 
        "humanoidmaze-medium-navigate": dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),

        "cube-triple-play":             dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.001),
        "cube-quadruple-play":          dict(mode="mse", inv_temp=10., guidance_coef=0.01, noisy_coef=0.1),
        "puzzle-4x4-play-sparse":       dict(mode="mse", inv_temp=10., guidance_coef=1.0,  noisy_coef=0.001),
        "antmaze-giant-navigate":       dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.001), 
        "humanoidmaze-large-navigate":  dict(mode="mse", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),
    },
    CGQL_LINEX={
        "puzzle-3x3-play-sparse":       dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.001),
        "scene-play-sparse":            dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),
        "cube-double-play":             dict(mode="linex", inv_temp=10., guidance_coef=0.01, noisy_coef=0.001),
        "antmaze-large-navigate":       dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.001), 
        "humanoidmaze-medium-navigate": dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),

        "cube-triple-play":             dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.001),
        "cube-quadruple-play":          dict(mode="linex", inv_temp=10., guidance_coef=0.01, noisy_coef=0.1),
        "puzzle-4x4-play-sparse":       dict(mode="linex", inv_temp=10., guidance_coef=1.0,  noisy_coef=0.001),
        "antmaze-giant-navigate":       dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.001), 
        "humanoidmaze-large-navigate":  dict(mode="linex", inv_temp=10., guidance_coef=0.1,  noisy_coef=0.1),
    },
    DSRL={
        "puzzle-3x3-play-sparse":       dict(noise_scale=1.0),
        "scene-play-sparse":            dict(noise_scale=0.4),
        "cube-double-play":             dict(noise_scale=1.0),
        "antmaze-large-navigate":       dict(noise_scale=0.8), 
        "humanoidmaze-medium-navigate": dict(noise_scale=0.6),

        "cube-triple-play":             dict(noise_scale=1.4),
        "cube-quadruple-play":          dict(noise_scale=1.4),
        "puzzle-4x4-play-sparse":       dict(noise_scale=1.0),
        "antmaze-giant-navigate":       dict(noise_scale=1.2), 
        "humanoidmaze-large-navigate":  dict(noise_scale=0.8),
    },
    FEDIT={
        "puzzle-3x3-play-sparse":       dict(edit_scale=0.2),
        "scene-play-sparse":            dict(edit_scale=0.2),
        "cube-double-play":             dict(edit_scale=0.2),
        "antmaze-large-navigate":       dict(edit_scale=0.2), 
        "humanoidmaze-medium-navigate": dict(edit_scale=0.5),

        "cube-triple-play":             dict(edit_scale=0.3),
        "cube-quadruple-play":          dict(edit_scale=0.4),
        "puzzle-4x4-play-sparse":       dict(edit_scale=0.8),
        "antmaze-giant-navigate":       dict(edit_scale=0.3), 
        "humanoidmaze-large-navigate":  dict(edit_scale=0.1),
    },
    FQL={
        "puzzle-3x3-play-sparse":       dict(alpha=300.0),
        "scene-play-sparse":            dict(alpha=300.0),
        "cube-double-play":             dict(alpha=300.0),
        "antmaze-large-navigate":       dict(alpha=3.0), 
        "humanoidmaze-medium-navigate": dict(alpha=30.0),

        "cube-triple-play":             dict(alpha=30.0),
        "cube-quadruple-play":          dict(alpha=100.0),
        "puzzle-4x4-play-sparse":       dict(alpha=1.0),
        "antmaze-giant-navigate":       dict(alpha=3.0), 
        "humanoidmaze-large-navigate":  dict(alpha=30.0),
    },
    FAWAC={
        "puzzle-3x3-play-sparse":       dict(inv_temp=0.8),
        "scene-play-sparse":            dict(inv_temp=6.4),
        "cube-double-play":             dict(inv_temp=0.8),
        "antmaze-large-navigate":       dict(inv_temp=6.4), 
        "humanoidmaze-medium-navigate": dict(inv_temp=6.4),

        "cube-triple-play":             dict(inv_temp=0.8),
        "cube-quadruple-play":          dict(inv_temp=0.8),
        "puzzle-4x4-play-sparse":       dict(inv_temp=0.8),
        "antmaze-giant-navigate":       dict(inv_temp=0.8), 
        "humanoidmaze-large-navigate":  dict(inv_temp=0.8),
    },
    QAM={
        "puzzle-3x3-play-sparse":       dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.),
        "scene-play-sparse":            dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.),
        "cube-double-play":             dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.),
        "antmaze-large-navigate":       dict(inv_temp=10., fql_alpha=0.,  edit_scale=0.), 
        "humanoidmaze-medium-navigate": dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.),

        "cube-triple-play":             dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.),
        "cube-quadruple-play":          dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.),
        "antmaze-giant-navigate":       dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.), 
        "humanoidmaze-large-navigate":  dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.),
        "puzzle-4x4-play-sparse":       dict(inv_temp=30., fql_alpha=0.,  edit_scale=0.),
    },
    QAM_EDIT={
        "puzzle-3x3-play-sparse":       dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.1),
        "scene-play-sparse":            dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.0),
        "cube-double-play":             dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.),
        "antmaze-large-navigate":       dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.1), 
        "humanoidmaze-medium-navigate": dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.1),

        "cube-triple-play":             dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.1),       
        "cube-quadruple-play":          dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.1),       
        "puzzle-4x4-play-sparse":       dict(inv_temp=0.1, fql_alpha=0.,  edit_scale=0.9),
        "antmaze-giant-navigate":       dict(inv_temp=10., fql_alpha=0.,  edit_scale=0.1),
        "humanoidmaze-large-navigate":  dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.1),
    },
    QAM_FQL={
        "puzzle-3x3-play-sparse":       dict(inv_temp=3.,  fql_alpha=0.,  edit_scale=0.),  
        "scene-play-sparse":            dict(inv_temp=1.,  fql_alpha=300.,edit_scale=0.),
        "cube-double-play":             dict(inv_temp=1.,  fql_alpha=0.,  edit_scale=0.),   
        "antmaze-large-navigate":       dict(inv_temp=3.,  fql_alpha=30., edit_scale=0.),
        "humanoidmaze-medium-navigate": dict(inv_temp=1.,  fql_alpha=30., edit_scale=0.),

        "cube-triple-play":             dict(inv_temp=10., fql_alpha=300.,edit_scale=0.),
        "cube-quadruple-play":          dict(inv_temp=0.3, fql_alpha=30., edit_scale=0.),
        "puzzle-4x4-play-sparse":       dict(inv_temp=3.,  fql_alpha=3.,  edit_scale=0.),
        "antmaze-giant-navigate":       dict(inv_temp=3.,  fql_alpha=30., edit_scale=0.),
        "humanoidmaze-large-navigate":  dict(inv_temp=0.3, fql_alpha=30., edit_scale=0.),
    },
    REBRAC={
        "puzzle-3x3-play-sparse":       dict(alpha_actor=0.1,  actor_noise=0.0),
        "scene-play-sparse":            dict(alpha_actor=0.03, actor_noise=0.0),
        "cube-double-play":             dict(alpha_actor=0.01, actor_noise=0.0),
        "antmaze-large-navigate":       dict(alpha_actor=0.01, actor_noise=0.0), 
        "humanoidmaze-medium-navigate": dict(alpha_actor=0.01, actor_noise=0.0),

        "cube-triple-play":             dict(alpha_actor=0.01, actor_noise=0.2),
        "cube-quadruple-play":          dict(alpha_actor=0.01, actor_noise=0.2),
        "puzzle-4x4-play-sparse":       dict(alpha_actor=0.01, actor_noise=0.2),
        "antmaze-giant-navigate":       dict(alpha_actor=0.01, actor_noise=0.0),
        "humanoidmaze-large-navigate":  dict(alpha_actor=0.01, actor_noise=0.1),
    },
    BAM={
        "puzzle-3x3-play-sparse":       dict(inv_temp=10.0),
        "scene-play-sparse":            dict(inv_temp=10.0),
        "cube-double-play":             dict(inv_temp=10.0),
        "antmaze-large-navigate":       dict(inv_temp=30.0), 
        "humanoidmaze-medium-navigate": dict(inv_temp=10.0),

        "cube-triple-play":             dict(inv_temp=30.0),
        "cube-quadruple-play":          dict(inv_temp=10.0),
        "antmaze-giant-navigate":       dict(inv_temp=30.0), 
        "humanoidmaze-large-navigate":  dict(inv_temp=30.0),
        "puzzle-4x4-play-sparse":       dict(inv_temp=30.0),
    },
    QSM={ 
        "puzzle-3x3-play-sparse":       dict(actor_loss_type="qsm", inv_temp=10., alpha=30., clip_sampler_before=False, clip_sampler_after=True),
        "scene-play-sparse":            dict(actor_loss_type="qsm", inv_temp=3.0, alpha=30., clip_sampler_before=False, clip_sampler_after=True),
        "cube-double-play":             dict(actor_loss_type="qsm", inv_temp=1.0, alpha=10., clip_sampler_before=False, clip_sampler_after=True),
        "antmaze-large-navigate":       dict(actor_loss_type="qsm", inv_temp=10., alpha=10., clip_sampler_before=False, clip_sampler_after=True),
        "humanoidmaze-medium-navigate": dict(actor_loss_type="qsm", inv_temp=10., alpha=30., clip_sampler_before=False, clip_sampler_after=True),

        "cube-triple-play":             dict(actor_loss_type="qsm", inv_temp=10., alpha=10., clip_sampler_before=False, clip_sampler_after=True),
        "cube-quadruple-play":          dict(actor_loss_type="qsm", inv_temp=1.0, alpha=10., clip_sampler_before=False, clip_sampler_after=True),
        "puzzle-4x4-play-sparse":       dict(actor_loss_type="qsm", inv_temp=10., alpha=1. , clip_sampler_before=False, clip_sampler_after=True),
        "antmaze-giant-navigate":       dict(actor_loss_type="qsm", inv_temp=10., alpha=10., clip_sampler_before=False, clip_sampler_after=True),
        "humanoidmaze-large-navigate":  dict(actor_loss_type="qsm", inv_temp=10., alpha=30., clip_sampler_before=False, clip_sampler_after=True),
    },
    DAC={ 
        "puzzle-3x3-play-sparse":       dict(actor_loss_type="dac", alpha=1.0, clip_sampler_before=True, clip_sampler_after=False),
        "scene-play-sparse":            dict(actor_loss_type="dac", alpha=1.0, clip_sampler_before=True, clip_sampler_after=False),
        "cube-double-play":             dict(actor_loss_type="dac", alpha=3.0, clip_sampler_before=True, clip_sampler_after=False),
        "antmaze-large-navigate":       dict(actor_loss_type="dac", alpha=0.3, clip_sampler_before=True, clip_sampler_after=False), 
        "humanoidmaze-medium-navigate": dict(actor_loss_type="dac", alpha=1.0, clip_sampler_before=True, clip_sampler_after=False),

        "cube-triple-play":             dict(actor_loss_type="dac", alpha=0.3, clip_sampler_before=True, clip_sampler_after=False),
        "cube-quadruple-play":          dict(actor_loss_type="dac", alpha=1.0, clip_sampler_before=True, clip_sampler_after=False),
        "puzzle-4x4-play-sparse":       dict(actor_loss_type="dac", alpha=1.0, clip_sampler_before=True, clip_sampler_after=False),
        "antmaze-giant-navigate":       dict(actor_loss_type="dac", alpha=0.3, clip_sampler_before=True, clip_sampler_after=False), 
        "humanoidmaze-large-navigate":  dict(actor_loss_type="dac", alpha=0.3, clip_sampler_before=True, clip_sampler_after=False), 
    },
)

agent_files = dict(
    FBRAC="agents/fbrac.py",
    IFQL="agents/ifql.py",
    CGQL="agents/cgql.py",
    CGQL_MSE="agents/cgql.py",
    CGQL_LINEX="agents/cgql.py",
    DSRL="agents/dsrl.py",
    FEDIT="agents/fedit.py",
    FQL="agents/fql.py",
    FAWAC="agents/fawac.py",
    BAM="agents/bam.py",
    QAM="agents/qam.py",
    QAM_EDIT="agents/qam.py",
    QAM_FQL="agents/qam.py",
    REBRAC="agents/rebrac.py",
    QSM="agents/dcgql.py",
    DAC="agents/dcgql.py",
)

methods = list(agent_files.keys())

print("# of methods:", len(agent_files))
print(methods)


for debug in [True, False]:
    gen = SbatchGenerator(j=num_jobs_per_gpu, limit=array_limit, prefix=("MUJOCO_GL=egl", "python main.py"))
    if debug:
        gen.add_common_prefix({"run_group": run_group + "_debug", "offline_steps": 100, "eval_episodes": 1, "eval_interval": 5, "start_training": 50, "online_steps": 100, "log_interval": 25})
    else:
        gen.add_common_prefix({"run_group": run_group, "offline_steps": 1000000, "online_steps": 500000, "save_interval": 50000, "eval_interval": 50000})

    env_names = []
    domains = []
    for domain in [
        "humanoidmaze-large-navigate",
        "puzzle-4x4-play-sparse",
        "humanoidmaze-medium-navigate",
        "antmaze-giant-navigate", 
        "cube-triple-play",
        "cube-quadruple-play",
        "antmaze-large-navigate", 
        "cube-double-play",
        "scene-play-sparse",
        "puzzle-3x3-play-sparse",
    ]:
        for task in [1, 2, 3, 4, 5]:
            if debug and task != 1: break
            if domain.endswith("-sparse"):
                name = domain[:-7]
            else:
                name = domain
            env_names.append(f"{name}-singletask-task{task}-v0")
            domains.append(domain)

    for seed in [10001, 20002, 30003, 40004, 50005, 60006, 70007, 80008, 90009, 100010, 110011, 120012]:
        if debug and seed != 10001: break
        for env_name, domain in zip(env_names, domains):
            
            if "ant" in domain or "humanoid" in domain:
                horizon_length = 1
            else:
                horizon_length = 5 # action chunking of length 5 for all manipulation tasks
            
            base_kwargs = {
                "seed": seed,
                "utd_ratio": 1, 
                'agent.num_qs': 10,
                "env_name": env_name,
                "sparse": False if "sparse" not in domain else True,
                "horizon_length": horizon_length,
                "agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
                "agent.action_chunking": True,
                "agent.actor_hidden_dims": '"(512, 512, 512, 512)"',
                "agent.value_hidden_dims": '"(512, 512, 512, 512)"',
                "agent.batch_size": 256,
                "agent.rho": 0.0 if "humanoid" in domain else 0.5,
            }
            
            if "cube-quadruple" in env_name:
                base_kwargs["ogbench_dataset_dir"] = data_root + "cube-quadruple-play-100m-v0/"
            if "puzzle-4x4" in env_name:
                base_kwargs["ogbench_dataset_dir"] = data_root + "puzzle-4x4-play-100m-v0/"
            
            for method in methods:
                kwargs = {"agent": agent_files[method], "tags": method, **base_kwargs}
                for k, v in agent_params[method][domain].items():
                    kwargs[f"agent.{k}"] = v

                gen.add_run(kwargs)

    sbatch_str_list = gen.generate_str()
    if debug:
        for index, sbatch_str in enumerate(sbatch_str_list):
            with open(f"sbatch/{run_group}-part{index+1}_debug.sh", "w") as f:
                f.write(sbatch_str)
    else:
        for index, sbatch_str in enumerate(sbatch_str_list):
            with open(f"sbatch/{run_group}-part{index+1}.sh", "w") as f:
                f.write(sbatch_str)
