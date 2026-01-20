
from generate import SbatchGenerator

run_group = "rlpd"
data_root = ... # the root directory of OGBench 100M datasets

num_jobs_per_gpu = 1
array_limit = 100

agent_params = dict(
    RLPD={ 
        "puzzle-3x3-play-sparse":       dict(bc_alpha=0.01),
        "scene-play-sparse":            dict(bc_alpha=0.3),
        "cube-double-play":             dict(bc_alpha=0.1),
        "antmaze-large-navigate":       dict(bc_alpha=0.0), 
        "humanoidmaze-medium-navigate": dict(bc_alpha=0.03),

        "cube-triple-play":             dict(bc_alpha=0.0),
        "cube-quadruple-play":          dict(bc_alpha=0.0),
        "puzzle-4x4-play-sparse":       dict(bc_alpha=0.0),
        "antmaze-giant-navigate":       dict(bc_alpha=0.01), 
        "humanoidmaze-large-navigate":  dict(bc_alpha=0.0), 
    },
)

for debug in [True, False]:
    gen = SbatchGenerator(j=num_jobs_per_gpu, limit=array_limit, prefix=("MUJOCO_GL=egl", "python main.py"))
    if debug:
        gen.add_common_prefix({"run_group": run_group + "_debug", "offline_steps": 0, "balanced_sampling": True, "eval_episodes": 1, "eval_interval": 5, "start_training": 50, "online_steps": 100, "log_interval": 25})
    else:
        gen.add_common_prefix({"run_group": run_group, "balanced_sampling": True, "offline_steps": 0, "online_steps": 500000, "save_interval": 50000, "eval_interval": 50000})
    
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
            
            kwargs = {"agent": "agents/rlpd.py", "tags": "RLPD", **base_kwargs}
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
