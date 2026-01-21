&nbsp;
<div id="user-content-toc" style="margin-bottom: 40px;margin-top: 60px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1> Q-learning with Adjoint Matching </h1>
      <br>
      <h2>[<a href="https://arxiv.org/pdf/2601.14234">Paper</a>]</h2>
       <!-- &emsp;|&emsp; [<a href="https://colinqiyangli.github.io/qam">Website</a>]</h2> -->
    </summary>
  </ul>
</div>


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./assets/title-light.png">
    <source media="(prefers-color-scheme: dark)" srcset="./assets/title-dark.png">
    <img alt="teaser figure" src="./assets/title-light.png" width="57.5%">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./assets/bar-light.png">
    <source media="(prefers-color-scheme: dark)" srcset="./assets/bar-dark.png">
    <img alt="bar plot" src="./assets/bar-light.png" width="41.8%">
  </picture>
</p>

## Overview
Q-learning with adjoint matching (QAM) uses [Adjoint Matching](https://arxiv.org/abs/2409.08861) to fine-tune a flow policy towards the optimal behavior-regularized solution efficiently!

Installation: `pip install -r requirements.txt`

## Reproducing paper results

We include the example command below for all three variants of our method on `cube-triple-task2`. We also release our experiment data at [exp_data/README.md](exp_data/README.md) and include some scripts for generating experiment commands in `experiments/*.py`. We hope this helps facilitate/speedup future research!

```bash
# QAM_EDIT
MUJOCO_GL=egl python main.py --run_group=reproduce --agent=agents/qam.py --tags=QAM_EDIT --seed=10001  --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --agent.inv_temp=3.0 --agent.fql_alpha=0.0 --agent.edit_scale=0.1

# QAM_FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent=agents/qam.py --tags=QAM_FQL --seed=10001 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --agent.inv_temp=10.0 --agent.fql_alpha=300.0 --agent.edit_scale=0.0

# QAM
MUJOCO_GL=egl python main.py --run_group=reproduce --agent=agents/qam.py --tags=QAM --seed=10001 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --agent.inv_temp=3.0 --agent.fql_alpha=0.0 --agent.edit_scale=0.0
```

## How do I obtain the 100M for puzzle-4x4 and cube-quadruple?
Please follow the instructions [here](https://github.com/seohongpark/horizon-reduction?tab=readme-ov-file#using-large-datasets) to obtain the large datasets.

## Acknowledgments
This codebase is built on top of [QC](https://github.com/colinqiyangli/qc).

## BibTeX
```
@article{li2026qam,
  author = {Qiyang Li and Sergey Levine},
  title  = {Q-learning with Adjoint Matching},
  conference = {arXiv Pre-print},
  year = {2026},
  url = {http://arxiv.org/abs/2601.14234},
}
```
