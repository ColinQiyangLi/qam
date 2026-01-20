
from generate import SbatchGenerator

run_group = "sensitivity"
data_root = ... # the root directory of OGBench 100M datasets

num_jobs_per_gpu = 1
array_limit = 100

agent_params = dict(
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
)

agent_files = dict(
    QAM_EDIT="agents/qam.py",
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
            if "maze" in domain:
                if task != 1:
                    continue
            else:
                if task != 2:
                    continue
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
            
            # NOCLIP
            for method in methods:
                kwargs = {"agent": agent_files[method], "agent.clip_grad": False, "tags": f"{method}-NOCLIP", **base_kwargs}
                for k, v in agent_params[method][domain].items():
                    kwargs[f"agent.{k}"] = v
                gen.add_run(kwargs)

            # Ensemble Size
            for method in methods:
                kwargs = {"agent": agent_files[method], "tags": f"{method}-ES=2", **base_kwargs}
                for k, v in agent_params[method][domain].items():
                    kwargs[f"agent.{k}"] = v
                kwargs["agent.num_qs"] = 2
                gen.add_run(kwargs)

            # Flow Steps
            for method in methods:
                for flow_steps in [1, 3, 20, 30]:
                    kwargs = {"agent": agent_files[method], "tags": f"{method}-T={flow_steps}", **base_kwargs}
                    for k, v in agent_params[method][domain].items():
                        kwargs[f"agent.{k}"] = v
                    kwargs["agent.flow_steps"] = flow_steps
                    gen.add_run(kwargs)

            # Inv Temp
            for method in methods:
                inv_temp = agent_params[method][domain]["inv_temp"]
                for name, inv_temp_t in zip(
                    ["d10", "d3", "m3", "m10"], 
                    [inv_temp / 10., inv_temp * 0.3, inv_temp * 3., inv_temp * 10.]
                ):
                    kwargs = {"agent": agent_files[method], "tags": f"{method}-{name}", **base_kwargs}
                    for k, v in agent_params[method][domain].items():
                        kwargs[f"agent.{k}"] = v
                    kwargs["agent.inv_temp"] = inv_temp_t
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
