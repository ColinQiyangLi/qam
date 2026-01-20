# Experiment Data

We release the experiment data to facilitate future research. Each file is a dictionary keyed by `(task_name/domain_name/all, method_name)` tuples. Each entry contains an numpy array of shape `(30, 12)` (except for RLPD where the shape is `(10, 12)`). The array stores the success rate at a regular interval of 50K training steps for 12 seeds.

See [sanity-check.ipynb](sanity-check.ipynb) for some quick examples for visualizing and plotting our experiment data.
