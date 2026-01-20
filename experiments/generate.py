import os

LIMIT = 800

class SbatchGenerator:
    def __init__(self, prefix=("MUJOCO_GL=egl", "python main.py"), j=1, limit=32, comment="default"):
        self.prefix = list(prefix)
        self.commands = []
        self.comment = comment
        self.j = j
        self.limit = limit

    def add_common_prefix(self, args):
        for key, value in args.items():
            self.prefix.append(f"--{key}={value}")

    def add_run(self, args):
        command_comps = []
        command_comps.extend(self.prefix)
        for key, value in args.items():
            command_comps.append(f"--{key}={value}")
        self.commands.append(" ".join(command_comps))

    def generate_str(self):

        num_jobs = len(self.commands)

        num_scripts = (num_jobs - 1) // LIMIT + 1
        sbatch_str_list = []
        for script_index in range(num_scripts):
            commands = self.commands[script_index * LIMIT: (script_index + 1) * LIMIT]
            num_jobs_partial = len(commands)

            num_arr = (num_jobs_partial - 1) // self.j + 1
            # print("\n".join(self.commands))

            path = "~" + os.getcwd()[len(os.path.expanduser("~")):]

            d_str = "\n  ".join(
                [
                    "[{}]='{}'".format(i + 1, command)
                    for i, command in enumerate(commands)
                ]
            )

            sbatch_str = f"""#!/bin/bash
#SBATCH --job-name=qam
#SBATCH --requeue
#SBATCH --array=1-{num_arr}%{self.limit}
#SBATCH --comment={self.comment}-part{script_index+1}

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={self.j}
JOB_N={num_jobs_partial}

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))

declare -a commands=(
  {d_str}
)

parallel --delay 5s --linebuffer -j {self.j} {{1}} ::: \"${{commands[@]:$COM_ID_S:$PARALLEL_N}}\"
            """
            sbatch_str_list.append(sbatch_str)
        
        return sbatch_str_list
